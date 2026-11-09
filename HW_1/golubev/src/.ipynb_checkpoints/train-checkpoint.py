import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from utils import (MODELS, SEED, TransformerClassifier, best_threshold, fit_with_threshold,
                   load_config, load_data, set_seed)


def config_defaults(cfg):
    data, cv = cfg.get('data', {}), cfg.get('cv', {})
    ev, tr = cfg.get('evaluate', {}), cfg.get('train', {})
    mapping = {'train': data.get('train'), 'test': data.get('test'), 'seed': cv.get('seed'),
               'results': ev.get('out'), 'models': tr.get('models'),
               'weights': tr.get('weights'), 'out': tr.get('out')}
    
    return {k: v for k, v in mapping.items() if v is not None}


def parse_args():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--config', help='путь к YAML-конфигу')
    known, _ = pre.parse_known_args()

    p = argparse.ArgumentParser(description='Обучение финальной модели', parents=[pre])
    p.add_argument('--train', default='data/train.csv')
    p.add_argument('--test', default='data/test.csv', help='пустая строка, чтобы не делать сабмит')
    p.add_argument('--models', nargs='+', choices=list(MODELS))
    p.add_argument('--weights', nargs='+', type=float,
                   help='веса моделей в ансамбле (по умолчанию равные)')
    p.add_argument('--results', default='results', help='папка с OOF из evaluate.py')
    p.add_argument('--out', default='artifacts')
    p.add_argument('--seed', type=int, default=SEED)
    p.set_defaults(**config_defaults(load_config(known.config)))
    args = p.parse_args()

    if not args.models:
        p.error('не заданы модели: укажите --models или train.models в конфиге')
    unknown = [m for m in args.models if m not in MODELS]
    if unknown:
        p.error(f'неизвестные модели в конфиге: {unknown}; доступны: {list(MODELS)}')

    return args


def save_model(model, name, out):
    if isinstance(model, TransformerClassifier):
        model.save(out / name)
    
    else:
        joblib.dump(model, out / f'{name}.joblib')


def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    weights = args.weights or [1 / len(args.models)] * len(args.models)
    if len(weights) != len(args.models):
        raise ValueError('число весов должно совпадать с числом моделей')
    
    weights = list(np.array(weights) / np.sum(weights))

    train = load_data(args.train)
    test = load_data(args.test) if args.test else None
    print(f'train: {len(train)} строк' + (f', test: {len(test)} строк' if test is not None else ''))

    if len(args.models) > 1:
        oof_paths = [Path(args.results) / f'oof_{m}.npy' for m in args.models]
        missing = [str(p) for p in oof_paths if not p.exists()]
        if missing:
            raise FileNotFoundError('нет OOF для подбора порога ансамбля: ' + ', '.join(missing)
                                    + '. Сначала запустите evaluate.py для этих моделей.')
        oof = sum(w * np.load(p) for w, p in zip(weights, oof_paths))
        threshold = best_threshold(train['target'].values, oof)

    test_proba = []
    for name in args.models:
        print(f'\nобучение {name} на всём train...')
        factory, tune = MODELS[name]
        model, thr = fit_with_threshold(factory, train, tune_threshold=tune and len(args.models) == 1)
        save_model(model, name, out)
        if len(args.models) == 1:
            threshold = thr

        if test is not None:
            test_proba.append(model.predict_proba(test))

    config = {'models': args.models, 'weights': weights, 'threshold': round(threshold, 2),
              'seed': args.seed}
    (out / 'config.json').write_text(json.dumps(config, indent=2, ensure_ascii=False))
    print(f'\nпорог: {threshold:.2f}; модели и config.json сохранены в {out}/')

    if test is not None:
        proba = sum(w * p for w, p in zip(weights, test_proba))
        sub = pd.DataFrame({'id': test['id'], 'target': (proba >= threshold).astype(int)})
        sub.to_csv(out / 'submission.csv', index=False)
        print(f'submission.csv: {len(sub)} строк, доля класса 1 = {sub["target"].mean():.3f}')


if __name__ == '__main__':
    main()
