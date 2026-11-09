import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from utils import (MODELS, N_SPLITS, SEED, compare_models, cv_evaluate, eval_oof,
                   load_config, load_data, make_folds, set_seed)


def config_defaults(cfg):
    """Переводит секции YAML-конфига в значения по умолчанию для argparse."""
    data, cv, ev = cfg.get('data', {}), cfg.get('cv', {}), cfg.get('evaluate', {})
    mapping = {'data': data.get('train'), 'n_splits': cv.get('n_splits'), 'seed': cv.get('seed'),
               'out': ev.get('out'), 'models': ev.get('models'), 'compare': ev.get('compare'),
               'blend': ev.get('blend'), 'weight': ev.get('weight')}
    
    return {k: v for k, v in mapping.items() if v is not None}


def parse_args():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--config', help='путь к YAML-конфигу')
    known, _ = pre.parse_known_args()

    p = argparse.ArgumentParser(description='Кросс-валидация моделей', parents=[pre])
    p.add_argument('--data', default='data/train.csv', help='путь к train.csv')
    p.add_argument('--out', default='results', help='папка для результатов')
    p.add_argument('--models', nargs='*', default=[], choices=list(MODELS),
                   help='модели для кросс-валидации')
    p.add_argument('--compare', nargs=2, metavar=('A', 'B'),
                   help='проверить гипотезу «B лучше A» по сохранённым результатам')
    p.add_argument('--blend', nargs=2, metavar=('A', 'B'),
                   help='оценить ансамбль двух моделей по сохранённым OOF')
    p.add_argument('--weight', type=float, default=0.7, help='вес модели A в ансамбле')
    p.add_argument('--n-splits', type=int, default=N_SPLITS)
    p.add_argument('--seed', type=int, default=SEED)
    p.set_defaults(**config_defaults(load_config(known.config)))
    args = p.parse_args()

    unknown = [m for m in args.models if m not in MODELS]
    if unknown:
        p.error(f'неизвестные модели в конфиге: {unknown}; доступны: {list(MODELS)}')

    return args


def load_result(out, name):
    path = out / f'cv_{name}.csv'
    if not path.exists():
        raise FileNotFoundError(f'{path} не найден: сначала запустите evaluate.py --models {name}')
    
    return pd.read_csv(path)


def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    df = load_data(args.data)
    folds = make_folds(df, n_splits=args.n_splits, seed=args.seed)
    print(f'train: {len(df)} строк, {args.n_splits} фолдов (seed={args.seed})')

    for name in args.models:
        factory, tune = MODELS[name]
        res, oof = cv_evaluate(factory, df, name, folds, tune_threshold=tune)
        res.to_csv(out / f'cv_{name}.csv', index=False)
        np.save(out / f'oof_{name}.npy', oof)

    if args.blend:
        a, b = args.blend
        oof_a, oof_b = np.load(out / f'oof_{a}.npy'), np.load(out / f'oof_{b}.npy')
        name = f'blend_{a}_{b}_w{args.weight}'
        res = eval_oof(args.weight * oof_a + (1 - args.weight) * oof_b, df, folds, name)
        res.to_csv(out / f'cv_{name}.csv', index=False)
        print(f'\n[{name}] f1 по фолдам:', res['f1'].round(3).tolist())

    files = sorted(out.glob('cv_*.csv'))
    if files:
        all_res = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        summary = (all_res.groupby('model')[['f1', 'precision', 'recall']]
                   .agg(['mean', 'std']).round(3)
                   .sort_values(('f1', 'mean'), ascending=False))
        summary.to_csv(out / 'summary.csv')
        print('\nСводка (F1 по классу 1, среднее и std по фолдам):')
        print(summary.to_string())

    if args.compare:
        a, b = args.compare
        r = compare_models(load_result(out, a), load_result(out, b))
        print(f'\nГипотеза: {b} лучше {a}')
        print(f'  F1: {a}={r["mean_a"]:.3f}, {b}={r["mean_b"]:.3f}, разница={r["mean_diff"]:+.3f}')
        print(f'  разброс между фолдами: {r["spread"]:.3f}')
        print(f'  выигрыш на каждом фолде: {r["wins_all_folds"]}')
        print(f'  гипотеза подтверждена: {r["confirmed"]}')


if __name__ == '__main__':
    main()
