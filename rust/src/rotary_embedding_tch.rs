use tch::{nn, Device, IndexOp, Kind, Tensor};
use std::f64::consts::PI;

pub fn broadcast(tensors: &Vec<Tensor>, dim: Option<&i64>) -> Tensor {
    let broadcasted_tensors = Tensor::broadcast_tensors(&tensors);
    Tensor::cat(&broadcasted_tensors, *dim.unwrap_or(&-1))
}

pub fn rotate_half(x: &Tensor) -> Tensor {
    let size = x.size();
    let d = size.last().unwrap() / 2;

    // Reshape x to split the last dimension into two dimensions (d and r)
    let x = x.view([-1, d as i64, 2]);

    // Unbind the last dimension into two separate tensors x1 and x2
    let unbound = x.unbind(-1);
    let (x1, x2) = (unbound[0].shallow_clone(), unbound[1].shallow_clone());

    // Stack -x2 and x1 along a new last dimension
    let x = Tensor::stack(&[-x2, x1], -1);

    // Flatten the last two dimensions back into a single dimension
    x.view([-1, (d * 2) as i64])
}

enum TensorOrScalar<'a> {
    Tensor(&'a Tensor),
    Scalar(&'a f64),
}

pub fn apply_rotary_emb(freqs: &Tensor, t: &Tensor, _start_index: Option<&i64>, _scale: Option<TensorOrScalar>, _seq_dim: Option<&i64>) -> Tensor {
    let start_index = _start_index.unwrap_or(&0);
    let scale = _scale.unwrap_or(TensorOrScalar::Scalar(&1.0));
    let seq_dim = _seq_dim.unwrap_or(&-2);

    let freqs_mut;
    let t = t.shallow_clone();

    if t.size().len() == 3 {
        let seq_len = t.size()[*seq_dim as usize].rem_euclid(t.dim().try_into().unwrap());
        freqs_mut = freqs.i((..seq_len, ..)).to_device(t.device());
    } else {
        freqs_mut = freqs.shallow_clone();
    }

    let binding = freqs_mut.size();
    let rot_dim = binding.last().unwrap();
    let end_index = start_index + rot_dim;

    assert!(*rot_dim <= *t.size().last().unwrap(), "feature dimension is not of sufficient size to rotate in all the positions");

    let t_left = t.i((.., ..*start_index));
    let mut t_middle = t.i((.., *start_index..end_index));
    let t_right = t.i((.., end_index..));

    match scale {
        TensorOrScalar::Tensor(scale) => {
            t_middle = (&t_middle * &freqs_mut.cos() * scale) + (&rotate_half(&t_middle) * &freqs_mut.sin() * scale);
        },
        TensorOrScalar::Scalar(scale) => {
            t_middle = (&t_middle * &freqs_mut.cos() * *scale) + (&rotate_half(&t_middle) * &freqs_mut.sin() * *scale);
        }
    }
    Tensor::cat(&[t_left, t_middle, t_right], -1)
}

pub fn apply_learned_rotations(rotations: &Tensor,  t: &Tensor, _start_index: Option<&i64>, _freq_ranges: Option<&Tensor>) -> Tensor {
    let start_index = _start_index.unwrap_or(&0);

    let mut rotations = rotations.shallow_clone();

    if _freq_ranges.is_some() {
        let freq_ranges = _freq_ranges.unwrap();
        let freq_ranges_broadcasted = freq_ranges.unsqueeze(-2).expand_as(&rotations);
        rotations = &rotations * &freq_ranges_broadcasted;

        let shape = rotations.size();
        let mut new_shape: Vec<i64> = shape[..shape.len() - 2].to_vec();
        new_shape.push(shape[shape.len() - 2] * shape[shape.len() - 1]);
        rotations = rotations.view(new_shape.as_slice());
    }

    const R: i64 = 2;
    let binding = rotations.size();
    let last_dim = binding.last().unwrap();
    let dim_minus_one = rotations.dim() - 1;
    let mut expanded_shape = rotations.size()[..dim_minus_one].to_vec();
    expanded_shape.push(last_dim * R as i64);

    rotations = rotations.unsqueeze(-1).expand(&expanded_shape, false);

    rotations = Tensor::cat(&[&rotations; R as usize], -1);

    apply_rotary_emb(&rotations, t, Some(start_index), None, None)
}

#[derive(Debug)]
pub struct RotaryEmbedding<'a> {
    varstore: &'a nn::Path<'a>,
    freqs_for: &'a str,
    cache_if_possible: bool,
    freqs: Tensor,
    learned_freq: bool,
    seq_before_head_dim: bool,
    default_seq_dim: i64,
    interpolate_factor: f64,
    use_xpos: bool,
    scale_base: i64,
}

impl RotaryEmbedding<'_> {
    pub fn new<'a>(
        varstore: &'a nn::Path<'a>,
        dim: i64,
        _custom_freqs: Option<&'a Tensor>,
        _freqs_for: Option<&'a str>,
        _theta: Option<&'a i64>,
        _max_freq: Option<&'a f64>,
        _num_freqs: Option<&'a i64>,
        _learned_freq: Option<&'a bool>,
        _use_xpos: Option<&'a bool>,
        _xpos_scale_base: Option<&'a i64>,
        _interpolate_factor: Option<&'a f64>,
        _theta_rescale_factor: Option<&'a f64>,
        _seq_before_head_dim: Option<&'a bool>,
        _cache_if_possible: Option<&'a bool>,
    ) -> RotaryEmbedding<'a> {
        let custom_freqs = None;
        let freqs_for = _freqs_for.unwrap_or("lang");
        let theta_rescale_factor: &f64 = _theta_rescale_factor.unwrap_or(&1.0);
        let theta = &(*_theta.unwrap_or(&10000) as f64) * theta_rescale_factor.powf((dim / (dim - 2)) as f64);
        let max_freq = _max_freq.unwrap_or(&10.0);
        let num_freqs = _num_freqs.unwrap_or(&1);
        let learned_freq = _learned_freq.unwrap_or(&false);
        let use_xpos = _use_xpos.unwrap_or(&false);
        let xpos_scale_base = _xpos_scale_base.unwrap_or(&512);
        let interpolate_factor = _interpolate_factor.unwrap_or(&1.0);
        let seq_before_head_dim = _seq_before_head_dim.unwrap_or(&false);
        let cache_if_possible = _cache_if_possible.unwrap_or(&true);

        let freqs_tensor;

        if _custom_freqs.is_none() {
            match _freqs_for {
                Some("pixel") => {
                    freqs_tensor = Tensor::linspace(1.0, max_freq / 2.0, dim / 2, (Kind::Float, varstore.device())) * PI;
                },
                Some("constant") => {
                    freqs_tensor = Tensor::ones(vec![*num_freqs], (Kind::Float, varstore.device()));
                },
                _ => {
                    let range = Tensor::arange_start_step(0, dim, 2, (Kind::Float, varstore.device()));
                    let sliced_scaled = &mut (range.i((..(dim / 2),)) / dim as f64);
                    freqs_tensor = 1.0 / Tensor::pow_(sliced_scaled, theta);
                }
            }
        } else {
            freqs_tensor = custom_freqs.unwrap();
        }

        let freqs = varstore.var_copy("freqs", &freqs_tensor);
        let default_seq_dim = if *seq_before_head_dim { -3 } else { -2 };

        assert!(*interpolate_factor >= 1.0, "interpolate factor must be greater than or equal to 1.0");

        let mut instance = RotaryEmbedding {
            varstore,
            freqs_for,
            cache_if_possible: *cache_if_possible,
            freqs,
            learned_freq: *learned_freq,
            seq_before_head_dim: *seq_before_head_dim,
            default_seq_dim: default_seq_dim,
            interpolate_factor: *interpolate_factor,
            use_xpos: *use_xpos,
            scale_base: *xpos_scale_base,
        };

        if !use_xpos {
            RotaryEmbedding::tmp_store(&mut instance, "scale", None);
        }

        RotaryEmbedding::tmp_store(&mut instance, "cached_freqs", None);
        RotaryEmbedding::tmp_store(&mut instance, "cached_scales", None);
        RotaryEmbedding::tmp_store(&mut instance, "dummy", Some(&Tensor::from(0)));

        let scale = (Tensor::arange_start_step(0, dim, 2, (Kind::Float, varstore.device())) + 0.4 * dim as f64) / (1.4 * dim as f64);
        RotaryEmbedding::tmp_store(&mut instance, "scale", Some(&scale));

        instance
    }

    fn tmp_store(&mut self, key: &str, value: Option<&Tensor>) -> Option<Tensor> {
        if value.is_none() {
            return None;
        }
        Some(self.varstore.var_copy(key, value.unwrap()))
    }

    fn get_seq_pos(&self, seq_len: &i64, kind: &Kind, device: &Device, _offset: Option<&i64>) -> Tensor {
        let offset = _offset.unwrap_or(&0);
        (Tensor::arange(*seq_len, (*kind, *device)) + *offset) / self.interpolate_factor
    }

    pub fn rotate_queries_or_keys(&mut self, t: &Tensor, _seq_dim: Option<&i64>, _offset: Option<&i64>, _freq_seq_len: Option<&i64>) -> Tensor {
        let offset = _offset.unwrap_or(&0);

        assert!(!self.use_xpos, "you must use `.rotate_queries_and_keys` method instead and pass in both queries and keys, for length extrapolatable rotary embeddings");

        let device = t.device();
        let kind = t.kind();
        let mut seq_len = t.size()[*_seq_dim.unwrap_or(&self.default_seq_dim) as usize];

        if _freq_seq_len.is_some() {
            let freq_seq_len = _freq_seq_len.unwrap();
            assert!(freq_seq_len >= &seq_len);
            seq_len = *freq_seq_len;
        }

        let t = &self.get_seq_pos(
            &seq_len,
            &kind,
            &device,
            Some(offset)
        );

        let mut freqs = self.forward(
            t,
            Some(&seq_len),
            Some(offset)
        );

        if _seq_dim.unwrap_or(&self.default_seq_dim) == &-3 {
            freqs = freqs.unsqueeze(1);
        }

        apply_rotary_emb(&freqs, t, None, None, Some(_seq_dim.unwrap_or(&self.default_seq_dim)))
    }

    pub fn rotate_queries_with_cached_keys(&mut self, q: &Tensor, k: &Tensor, _seq_dim: Option<&i64>, _offset: Option<&i64>) -> (Tensor, Tensor) {
        let default_seq_dim = self.default_seq_dim;
        let seq_dim = _seq_dim.unwrap_or(&default_seq_dim);

        let q_len = q.size()[*seq_dim as usize];
        let k_len = k.size()[*seq_dim as usize];

        assert!(q_len <= k_len, "query length must be less than or equal to key length");

        let rotated_q = self.rotate_queries_or_keys(q, Some(seq_dim), None, Some(&k_len));
        let rotated_k = self.rotate_queries_or_keys(k, Some(seq_dim), None, None);

        (rotated_q, rotated_k)
    }

    pub fn rotate_queries_and_keys(&mut self, q: &Tensor, k: &Tensor, _seq_dim: Option<&i64>) -> (Tensor, Tensor) {
        let default_seq_dim = self.default_seq_dim;
        let seq_dim = _seq_dim.unwrap_or(&default_seq_dim);

        assert!(self.use_xpos, "you must use `.rotate_queries_or_keys` method instead and pass in only queries, for non-length extrapolatable rotary embeddings");
        let device = q.device();
        let kind = q.kind();
        let seq_len = q.size()[*seq_dim as usize];

        let seq = self.get_seq_pos(&seq_len, &kind, &device, None);

        let mut freqs = self.forward(&seq, Some(&seq_len), None);
        let mut scale = self.get_scale(&seq, Some(&seq_len), None).to_kind(kind);

        if *seq_dim == -3 {
            freqs = freqs.unsqueeze(1);
            scale = scale.unsqueeze(1);
        }

        let mut rotated_q = apply_rotary_emb(&freqs, q, None, Some(TensorOrScalar::Tensor(&scale)), Some(seq_dim));
        let mut rotated_k = apply_rotary_emb(&freqs, k, None, Some(TensorOrScalar::Tensor(&scale.pow_(-1))), Some(seq_dim));

        rotated_q = rotated_q.to_kind(q.kind());
        rotated_k = rotated_k.to_kind(k.kind());

        (rotated_q, rotated_k)
    }

    pub fn get_scale(&mut self, t: &Tensor, _seq_len: Option<&i64>, _offset: Option<&i64>) -> Tensor {
        assert!(self.use_xpos, "you must use `.rotate_queries_or_keys` method instead and pass in only queries, for non-length extrapolatable rotary embeddings");

        let offset = _offset.unwrap_or(&0);

        let seq_len = _seq_len.unwrap_or(&0);
        let should_cache = self.cache_if_possible && _seq_len.is_some();

        let _cached_scales = self.varstore.get("cached_scales");

        if should_cache && _cached_scales.is_some() {
            let cached_scales = _cached_scales.unwrap();
            if (seq_len + offset) <= cached_scales.size()[0 as usize] {
                return cached_scales.i(*offset..(offset + seq_len));
            }
        }

        let mut scale = Tensor::from(1.0);
        if self.use_xpos {
            let power = (t - t.size()[0] / 2) / self.scale_base;
            scale = self.varstore.get("scale").unwrap().pow(&power.unsqueeze(-1));
            scale = Tensor::cat(&vec![&scale, &scale], -1);
        }

        if should_cache {
            self.tmp_store("cached_scales", Some(&scale));
        }

        scale
    }

    pub fn get_axial_freqs(&mut self, dims: &[i64]) -> Tensor {
        let mut all_freqs = Vec::new();

        for (ind, &dim) in dims.iter().enumerate() {
            let pos = if self.freqs_for == "pixel" {
                Tensor::linspace(-1.0, 1.0, dim, (Kind::Float, self.varstore.device()))
            } else {
                Tensor::arange(dim, (Kind::Int64, self.varstore.device()))
            };

            let freqs = self.forward(&pos, Some(&dim), None);

            // Construct new shape for freqs with additional dimensions
            let mut new_shape = vec![1; dims.len()];
            new_shape[ind] = -1;
            let freqs_reshaped = freqs.view(new_shape.as_slice());
            all_freqs.push(freqs_reshaped);
        }

        let broadcasted_freqs = Tensor::broadcast_tensors(&all_freqs);
        Tensor::cat(&broadcasted_freqs, -1)
    }
}

trait ModuleTII {
    fn forward(&mut self, t: &Tensor, _seq_len: Option<&i64>, _offset: Option<&i64>) -> Tensor;
}

impl ModuleTII for RotaryEmbedding<'_> {
    fn forward(&mut self, t: &Tensor, _seq_len: Option<&i64>, _offset: Option<&i64>) -> Tensor {
        let offset = _offset.unwrap_or(&0);

        let should_cache = self.cache_if_possible && !self.learned_freq && _seq_len.is_some() && self.freqs_for != "pixel";

        if should_cache {
            let seq_len = _seq_len.unwrap();
            let _cached_freqs = self.varstore.get("cached_freqs");
            if _cached_freqs.is_some() {
                let cached_freqs = _cached_freqs.unwrap();
                if (offset + seq_len) <= cached_freqs.size()[0] {
                    return cached_freqs.i(*offset..(offset + seq_len)).detach();
                }
            }
        }

        let freqs = self.freqs.shallow_clone();

        // Emulate einsum operation
        let t_broadcasted = t.to_kind(freqs.kind()).unsqueeze(-1).expand_as(&freqs);
        let freqs_einsum = &t_broadcasted * &freqs;

        // Emulate repeat operation
        const R: i64 = 2;
        let freqs_expanded = freqs_einsum.unsqueeze(-1).expand(&[-1, -1, R], false);
        let freqs_repeated = Tensor::cat(&[&freqs_expanded; R as usize], -1);

        if should_cache {
            self.tmp_store("cached_freqs", Some(&freqs_repeated.detach()));
        }

        freqs
    }
}
