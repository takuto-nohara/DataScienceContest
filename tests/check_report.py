"""Independently recompute MATLAB report metrics using only Python's stdlib."""
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def records(value):
    return value if isinstance(value, list) else [value]


def check(folder):
    report = json.loads((folder / 'results.json').read_text(encoding='utf-8-sig'))
    board = {row['model']: row for row in records(report['leaderboard'])}
    details = {row['model']: row for row in records(report['details'])}
    for name, row in board.items():
        classes = details[name]['test']['classes']
        with (folder / f'{name}_predictions.csv').open(encoding='utf-8-sig', newline='') as file:
            predictions = list(csv.DictReader(file))
        assert [int(p['row']) for p in predictions] == list(range(1, len(predictions) + 1))
        matrix = [[sum(p['truth'] == a and p['prediction'] == b for p in predictions)
                   for b in classes] for a in classes]
        assert matrix == details[name]['test']['confusion']
        correct = sum(matrix[i][i] for i in range(len(classes)))
        assert correct == row['testCorrect'] and len(predictions) == row['testTotal']
        assert math.isclose(correct / len(predictions), row['testAccuracy'], abs_tol=1e-12)
        scores = []
        for i in range(len(classes)):
            tp = matrix[i][i]
            denominator = sum(matrix[i]) + sum(r[i] for r in matrix)
            scores.append(2 * tp / denominator if denominator else 0)
        assert math.isclose(sum(scores) / len(scores), row['testMacroF1'], abs_tol=1e-12)
        manifest = json.loads((ROOT / 'model' / name / 'manifest.json').read_text())
        assert hashlib.sha256((ROOT / 'model' / name / 'trained.mat').read_bytes()).hexdigest() == manifest['artifactSHA256']
        protocol = json.loads((ROOT / 'evaluation' / 'protocol.json').read_text())
        assert set(manifest['fitRows']).issubset(protocol['fitRows'])
        assert set(manifest['fitRows']).isdisjoint(protocol['validationRows'])
    return board


if __name__ == '__main__':
    provenance = json.loads((ROOT / 'data' / 'manifest.json').read_text(encoding='utf-8-sig'))
    for entry in provenance['files']:
        assert hashlib.sha256((ROOT / entry['path']).read_bytes()).hexdigest() == entry['sha256']
    all_models = check(ROOT / 'evaluation' / 'latest')
    subset = check(ROOT / 'evaluation' / 'subset_check')
    assert len(all_models) > 2 and len(subset) == 3
    for name, row in subset.items():
        for metric in ('validationAccuracy', 'testAccuracy', 'testMacroF1', 'testCorrect'):
            assert row[metric] == all_models[name][metric], (name, metric)
    refinement = ROOT / 'evaluation' / 'refinement_comparison'
    if refinement.exists():
        refined = check(refinement)
        final = refined['proto_v10']
        assert final['testCorrect'] == final['testTotal'] == 201
        assert final['testAccuracy'] == final['testMacroF1'] == 1
        report = json.loads((refinement / 'results.json').read_text())
        assert report['containsTestAwareModels'] is True
        assert final['evaluationRole'] == 'development_tuned'
        for name, row in all_models.items():
            for metric in ('testCorrect', 'testAccuracy', 'validationAccuracy'):
                assert refined[name][metric] == row[metric]
        print('REFINEMENT_REPORT_CHECKS_PASSED: 201/201, F1=1, development label, baseline scores unchanged')
    print(f'REPORT_CHECKS_PASSED: {len(all_models)} models, 3-model subset, independent accuracy/F1/confusion calculations, model checksums')
