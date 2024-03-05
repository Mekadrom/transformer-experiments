use anyhow::Result;
use tch::{nn, Device};

mod rotary_embedding_tch;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let varstore = nn::VarStore::new(Device::cuda_if_available());
    let rotary_embedding_torch = rotary_embedding_tch::RotaryEmbedding::new(
        &varstore.root(),
        64,
        None, None, None, None, None, None, None, None, None, None, None, None
    );
    Ok(())
}
