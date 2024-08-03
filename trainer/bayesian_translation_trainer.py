from skopt import Optimizer
from skopt.space import Real, Integer, Categorical

import os
import trainer.translation_trainer as translation_trainer
import utils
import yaml_dict

class MetricDef:
    def __init__(self, name, goal, weight, normalize):
        self.name = name
        self.goal = goal
        self.weight = weight
        self.normalize = normalize

    def calc(self, value):
        return (value / self.normalize) * self.weight
    
    def __str__(self):
        return f"MetricDef(name={self.name}, goal={self.goal}, weight={self.weight}, normalize={self.normalize})"
    
    def __repr__(self):
        return self.__str__()

class BayesianIter:
    def __init__(self, args):
        self.args = args

        self.space = [self.get_arg_def(arg) for arg in args.bayesian_arg_ranges]
        self.metrics = [self.get_metric_def(metric) for metric in args.bayesian_metrics]

        print(f"space: {[s.name for s in self.space]}")
        print(f"metrics: {self.metrics}")

        self.optimizer = Optimizer(
            dimensions=self.space,
            base_estimator='GP',
            n_initial_points=1,
        )

    def get_arg_def(self, arg):
        if arg['type'] == 'int':
            return Integer(arg['min'], arg['max'], name=arg['name'])
        elif arg['type'] == 'float':
            return Real(arg['min'], arg['max'], name=arg['name'])
        elif arg['type'] == 'categorical':
            return Categorical(arg['values'], name=arg['name'])
        else:
            raise ValueError(f"unknown arg type {arg['type']}")
        
    def get_metric_def(self, metric):
        return MetricDef(metric['name'], metric['goal'], float(metric['weight']), float(metric['normalize']))

    def train(self):
        default_args = utils.load_yaml(os.path.join('translation', 'configs', 'default.yaml'), None)

        print(f"initial_run_metrics: {self.args.initial_run_metrics}")
        print(f"initial_run_args: {list(self.args.initial_run_args.values())}")

        sacrebleu, time_taken, param_count = self.args.initial_run_metrics['sacrebleu'], self.args.initial_run_metrics['time_taken'], self.args.initial_run_metrics['param_count']
        metric_score = self.score_metrics(sacrebleu=sacrebleu, time_taken=time_taken, param_count=param_count, invalid_run=False)

        self.optimizer.tell(list(self.args.initial_run_args.values()), metric_score)

        for bayesian_iter in range(self.args.bayesian_iter_count):
            next_args = {}
            next_args.update(default_args)

            ask = self.optimizer.ask(n_points=1, strategy='cl_min')[0]

            ask = {self.space[i].name: ask for i, ask in enumerate(ask)}

            print(f"ask: {ask}")

            next_args.update(ask)

            next_args = yaml_dict.YamlDict(next_args)
            next_args['run_name'] = f"{self.args.run_name}/bayesian_{bayesian_iter}"
            next_args['tokenizer_run_name'] = self.args.tokenizer_run_name
            next_args['lr'] = utils.get_lr(step=1, d_model=next_args.d_model, warmup_steps=next_args.warmup_steps)
            if hasattr(next_args, 'tokens_in_batch'):
                setattr(next_args, 'batches_per_step', next_args.target_tokens_per_batch // next_args.tokens_in_batch)

            print(f"next_args: {next_args}")

            trainer = translation_trainer.TranslationTrainer(next_args)
            if self.is_valid(utils.YamlDict(ask), trainer.model):
                sacrebleu, time_taken, param_count = trainer.train()
                # sacrebleu, time_taken, param_count = 0, 0, 0 # for testing

                print(f"bayesian_iter {bayesian_iter} sacrebleu: {sacrebleu}")

                metric_score = self.score_metrics(sacrebleu=sacrebleu, time_taken=time_taken, param_count=param_count, invalid_run=False)
            else:
                sacrebleu, time_taken, param_count = 0, 0, 0
                metric_score = self.score_metrics(sacrebleu=sacrebleu, time_taken=time_taken, param_count=param_count, invalid_run=True)
                print(f"bayesian_iter {bayesian_iter} invalid run")

            self.optimizer.tell(list(ask.values()), metric_score)

    def is_valid(self, args, model):
        if utils.count_parameters(model) > self.args.bayesian_param_count_limit:
            print(f"invalid param count {utils.count_parameters(model)}")
            return False
        
        if args.d_inner < args.d_model:
            print(f"d_inner < d_model {args.d_inner} < {args.d_model}")
            return False
        
        if args.positional_encoding_dim > args.d_queries:
            print(f"positional_encoding_dim > d_queries {args.positional_encoding_dim} > {args.d_queries}")
            return False
        
        if args.m_encoder_independent_layers > args.n_encoder_layers:
            print(f"m_encoder_independent_layers > n_encoder_layers {args.m_encoder_independent_layers} > {args.n_encoder_layers}")
            return False
        
        if args.m_decoder_independent_layers > args.n_decoder_layers:
            print(f"m_decoder_independent_layers > n_decoder_layers {args.m_decoder_independent_layers} > {args.n_decoder_layers}")
            return False
        
        if args.moe_top_k > args.moe_n_experts:
            print(f"moe_top_k > moe_n_experts {args.moe_top_k} > {args.moe_n_experts}")
            return False

        if args.m_encoder_independent_layers > 0:
            setattr(args, 'encoder_param_sharing_type', 'cycle-rev')
            print(f"encoder_param_sharing_type: cycle-rev")

        if args.m_decoder_independent_layers > 0:
            setattr(args, 'decoder_param_sharing_type', 'cycle-rev')
            print(f"decoder_param_sharing_type: cycle-rev")

        setattr(args, 'use_admin', bool(args.use_admin))
        setattr(args, 'learnable_positional_encoding', bool(args.use_admin))

        return True
        
    def score_metrics(self, **kwargs):
        score = 0

        for kwarg in kwargs:
            metric = next((metric for metric in self.metrics if metric.name == kwarg), None)

            if metric is not None:
                score += metric.calc(kwargs[kwarg])

        return score
