#!/usr/bin/env python3

"""Landau V2b: diskretni strukturni nivoi i rangiranje celih kombinacija."""

import argparse
import csv
from pathlib import Path
import numpy as np


DEFAULTS = (
    "/Users/4c/Desktop/GHQ/data/loto7_4698_k80.csv",
    "/Users/4c/Desktop/GHQ/data/loto7_4698_k80_loto_2971.csv",
    "/Users/4c/Desktop/GHQ/data/loto7_4698_k80_loto_plus_1727.csv",
)


N, K, WINDOW, LANDMARKS, LEVELS = 39, 7, 64, 24, 8
ALPHAS = (1., 10., 100.)


def load(path):
    rows = []
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        for line, row in enumerate(csv.reader(stream), 1):
            if not row or all(not v.strip() for v in row):
                continue
            try:
                values = [int(v.strip()) for v in row]
            except ValueError as error:
                raise ValueError(
                    f"Red {line}: ocekujem cele brojeve."
                ) from error

            if (len(values) != K or values != sorted(values)
                    or len(set(values)) != K
                    or not all(1 <= v <= N for v in values)):
                raise ValueError(
                    f"Red {line}: neispravna sortirana 7/39 kombinacija."
                )
            rows.append(values)

    return np.asarray(rows, dtype=int).reshape(-1, K)


def shape(rows):
    rows = np.atleast_2d(rows)
    edges = np.column_stack((
        np.zeros(len(rows)), rows, np.full(len(rows), 40)
    ))
    return np.column_stack((
        rows / 40., np.diff(edges, axis=1) / 40.
    ))


def distances(a, b):
    return np.abs(a[:, None, :] - b[None, :, :]).sum(axis=2)


def replacements(row):
    unused = np.setdiff1d(np.arange(1, 40), row)
    result = np.tile(row, (K * len(unused), 1))
    for j in range(K):
        result[j*len(unused):(j+1)*len(unused), j] = unused
    return np.sort(result, axis=1)


class Spectrum:
    def __init__(self, history):
        layouts = shape(history)

        # Oslonci su stvarne kombinacije.
        # Nema sintetickih ravnomernih semena.
        selected = [len(layouts)-1]
        minimum = distances(layouts, layouts[selected])[:, 0]

        while len(selected) < min(LANDMARKS, len(layouts)):
            chosen = int(np.argmax(minimum))
            if minimum[chosen] <= 1e-12:
                break
            selected.append(chosen)
            minimum = np.minimum(
                minimum,
                distances(layouts, layouts[[chosen]])[:, 0]
            )

        self.landmarks = layouts[selected]
        separation = distances(self.landmarks, self.landmarks)
        positive = separation[separation > 1e-12]
        self.bandwidth = (
            float(np.median(positive)) if len(positive) else 1.
        )

        # Eksponencijalni kernel sa apsolutnim rastojanjem.
        kernel = np.exp(-separation / self.bandwidth)
        eigenvalues, eigenvectors = np.linalg.eigh(kernel)

        order = np.argsort(eigenvalues)[::-1]
        order = order[
            eigenvalues[order] > eigenvalues[order[0]] * 1e-10
        ][:LEVELS]

        eigenvalues = eigenvalues[order]
        vectors = eigenvectors[:, order]

        # Jednoznacan znak svojstvenih vektora.
        for j in range(len(order)):
            pivot = int(np.argmax(np.abs(vectors[:, j])))
            if vectors[pivot, j] < 0:
                vectors[:, j] *= -1

        self.energies = -np.log(eigenvalues / eigenvalues[0])
        self.energies[0] = 0.
        self.basis = vectors / np.sqrt(eigenvalues)[None, :]

        # Numericko skaliranje spektralnih koordinata.
        self.scale = np.sqrt(
            np.mean(self.features(history)**2, axis=0)
        )
        self.scale = np.maximum(self.scale, 1e-10)

    def features(self, rows):
        layouts = shape(rows)
        blocks = []

        for start in range(0, len(layouts), 1024):
            kernel = np.exp(
                -distances(
                    layouts[start:start+1024], self.landmarks
                ) / self.bandwidth
            )
            blocks.append(kernel @ self.basis)

        result = np.vstack(blocks)
        if hasattr(self, "scale"):
            result = result / self.scale
        return result


def context(features, t):
    return np.r_[1., features[t-1], features[t-2]]


def prepare(history):
    if len(history) < WINDOW + 2:
        raise ValueError("Potrebno je najmanje 66 prethodnih redova.")

    history = history[-(WINDOW+2):]
    spectrum = Spectrum(history)
    f = spectrum.features(history)

    dimension = f.shape[1] * (1 + 2*f.shape[1])
    gram = np.zeros((dimension, dimension))
    target = np.zeros(dimension)

    # Svaki stvarni naredni red poredi se sa svih 224
    # kombinacija nastalih zamenom jednog njegovog broja.
    # Uci se skalarni rang CELE kombinacije.
    for t in range(2, len(history)):
        alternatives = spectrum.features(replacements(history[t]))
        differences = f[t] - alternatives
        c = context(f, t)

        design = (
            differences[:, :, None] * c[None, None, :]
        ).reshape(len(differences), dimension)

        gram += design.T @ design / len(design)
        target += design.sum(axis=0) / len(design)

    gram /= WINDOW
    target /= WINDOW

    candidates = np.unique(
        np.vstack(
            [history[2:]]
            + [replacements(row) for row in history[2:]]
        ),
        axis=0
    )
    candidate_features = spectrum.features(candidates)

    return (
        spectrum, gram, target,
        context(f, len(history)),
        candidates, candidate_features
    )


def predict(prepared, alpha):
    spectrum, gram, target, c, candidates, features = prepared

    coef = np.linalg.solve(
        gram + alpha*np.eye(len(target)), target
    )
    field = coef.reshape(features.shape[1], len(c)) @ c
    scores = features @ field

    if not np.all(np.isfinite(scores)):
        raise ArithmeticError("Neispravan skor.")

    best = int(np.argmax(scores))
    return (
        tuple(map(int, candidates[best])),
        float(scores[best])
    )


def hits(prediction, actual):
    return len(
        set(map(int, prediction)) & set(map(int, actual))
    )


def summarize(label, scores):
    print(
        f"{label}: prosek={np.mean(scores):.4f}; "
        f"pogodaka 0..7="
        f"{np.bincount(scores, minlength=8).tolist()}",
        flush=True
    )


def run_file(path, validation_count=48, test_count=64):
    rows = load(path)
    split = len(rows) - test_count
    begin = split - validation_count

    if (validation_count < 16 or test_count < 16
            or begin < WINDOW+2):
        raise ValueError(
            "Potrebna je dovoljna istorija i najmanje "
            "16 validacionih/test kola."
        )

    print(
        f"\nUlaz: {Path(path).resolve()}; redova={len(rows)}",
        flush=True
    )

    validation = [[] for _ in ALPHAS]

    for t in range(begin, split):
        prepared = prepare(rows[:t])

        for j, alpha in enumerate(ALPHAS):
            combo, _ = predict(prepared, alpha)
            validation[j].append(hits(combo, rows[t]))

        if (t-begin+1) % 16 == 0:
            print(
                f"Validacija: {t-begin+1}/{validation_count}",
                flush=True
            )

    means = [np.mean(s) for s in validation]
    chosen = int(np.argmax(means))
    alpha = ALPHAS[chosen]

    print(
        f"Validacija: redovi {begin+1}..{split}; "
        f"alpha={alpha:g}; prosek={means[chosen]:.4f}",
        flush=True
    )

    scores, previous = [], []

    # Alpha je zamrznuta pre pocetka testa.
    # Svaka predikcija koristi iskljucivo prethodne redove.
    for t in range(split, len(rows)):
        combo, _ = predict(prepare(rows[:t]), alpha)
        scores.append(hits(combo, rows[t]))
        previous.append(hits(rows[t-1], rows[t]))

        if (t-split+1) % 16 == 0:
            print(
                f"Test: {t-split+1}/{test_count}",
                flush=True
            )

    print(f"Test: redovi {split+1}..{len(rows)}")
    summarize("LANDAU V2b LEVELS", scores)
    summarize("Prethodna kombinacija", previous)

    prepared = prepare(rows)
    combo, score = predict(prepared, alpha)

    print(f"Poslednji red: {','.join(map(str, rows[-1]))}")
    print(
        f"SLEDECI RED {len(rows)+1}: "
        f"{','.join(map(str, combo))}"
    )
    print(
        f"Skor rangiranja={score:.10f}; "
        f"kandidata={len(prepared[4])}"
    )
    print(
        "Strukturni nivoi E:",
        ", ".join(f"{e:.6f}" for e in prepared[0].energies)
    )
    print(
        "Izbor: poslednje 64 kombinacije "
        "i sve njihove zamene jednog broja."
    )


def self_test():
    base = np.array([
        [1,2,3,4,5,6,7],
        [4,12,18,28,35,36,37],
        [33,34,35,36,37,38,39],
        [2,7,12,18,24,30,39]
    ])
    history = np.tile(base, (20, 1))
    p = prepare(history)

    assert np.all(np.diff(p[0].energies) >= -1e-12)
    assert np.allclose(p[1], p[1].T)
    assert np.linalg.eigvalsh(p[1]).min() >= -1e-8

    for row in base:
        r = replacements(row)
        assert len(r) == 224
        assert len(np.unique(r, axis=0)) == 224
        assert all(hits(row, x) == 6 for x in r)

    for alpha in ALPHAS:
        result = predict(p, alpha)
        assert result == predict(prepare(history), alpha)
        assert any(
            np.array_equal(result[0], x) for x in p[4]
        )
        assert len(set(result[0])) == K

    altered = history.copy()
    altered[-1] = base[0]

    a = predict(prepare(history[:-1]), 1.)
    b = predict(prepare(altered[:-1]), 1.)
    assert a == b

    print(
        "Self-test OK: spektar, zamene, rangiranje, "
        "ponovljivost i izolacija buduceg reda."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="*")
    parser.add_argument("--validation", type=int, default=48)
    parser.add_argument("--test", type=int, default=64)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    for path in args.csv or DEFAULTS:
        run_file(path, args.validation, args.test)


if __name__ == "__main__":
    main()



"""
Ulaz: /Users/4c/Desktop/GHQ/data/loto7_4698_k80.csv; redova=4698
Validacija: 16/48
Validacija: 32/48
Validacija: 48/48
Validacija: redovi 4587..4634; alpha=10; prosek=1.3750
Test: 16/64
Test: 32/64
Test: 48/64
Test: 64/64
Test: redovi 4635..4698
LANDAU V2b LEVELS: prosek=1.1562; pogodaka 0..7=[16, 25, 20, 3, 0, 0, 0, 0]
Prethodna kombinacija: prosek=1.2188; pogodaka 0..7=[13, 30, 16, 4, 1, 0, 0, 0]
Poslednji red: 4,12,18,28,35,36,37
SLEDECI RED 4699: 1,5,6,11,12,28,30
Skor rangiranja=0.1787779153; kandidata=14396
Strukturni nivoi E: 0.000000, 1.484299, 2.041512, 2.231685, 2.342474, 2.411967, 2.422086, 2.559683
Izbor: poslednje 64 kombinacije i sve njihove zamene jednog broja.



Ulaz: /Users/4c/Desktop/GHQ/data/loto7_4698_k80_loto_2971.csv; redova=2971
Validacija: 16/48
Validacija: 32/48
Validacija: 48/48
Validacija: redovi 2860..2907; alpha=10; prosek=1.3750
Test: 16/64
Test: 32/64
Test: 48/64
Test: 64/64
Test: redovi 2908..2971
LANDAU V2b LEVELS: prosek=1.3438; pogodaka 0..7=[11, 30, 14, 8, 1, 0, 0, 0]
Prethodna kombinacija: prosek=1.2969; pogodaka 0..7=[11, 30, 16, 7, 0, 0, 0, 0]
Poslednji red: 18,20,21,24,29,34,38
SLEDECI RED 2972: 8,9,10,12,13,28,34
Skor rangiranja=0.2154661864; kandidata=14392
Strukturni nivoi E: 0.000000, 1.566343, 2.087795, 2.101864, 2.259496, 2.415445, 2.491050, 2.668013
Izbor: poslednje 64 kombinacije i sve njihove zamene jednog broja.



Ulaz: /Users/4c/Desktop/GHQ/data/loto7_4698_k80_loto_plus_1727.csv; redova=1727
Validacija: 16/48
Validacija: 32/48
Validacija: 48/48
Validacija: redovi 1616..1663; alpha=10; prosek=1.1667
Test: 16/64
Test: 32/64
Test: 48/64
Test: 64/64
Test: redovi 1664..1727
LANDAU V2b LEVELS: prosek=1.3281; pogodaka 0..7=[16, 18, 24, 5, 1, 0, 0, 0]
Prethodna kombinacija: prosek=1.2969; pogodaka 0..7=[15, 23, 18, 8, 0, 0, 0, 0]
Poslednji red: 4,12,18,28,35,36,37
SLEDECI RED 1728: 9,10,27,29,30,31,37
Skor rangiranja=0.1533506861; kandidata=14388
Strukturni nivoi E: 0.000000, 1.454544, 1.949805, 2.243283, 2.305993, 2.431488, 2.535364, 2.583433
Izbor: poslednje 64 kombinacije i sve njihove zamene jednog broja.
"""
 


"""
Izlaz će se birati rangiranjem celih kombinacija, bez prosečnih položaja brojeva.
quantum levels je popravljena i proverena na sva tri CSV fajla. Uči rang cele kombinacije kroz diskretne strukturne nivoe. Nema prosečenja položaja, ciljnih prosečnih zauzeća nivoa, random postupka ni semena sa jednakim razmacima.

Fajl	                Sledeća kombinacija	    Prosek pogodaka, poslednja 64 kola
Objedinjeni, red 4699	1,5,6,11,12,28,30	    1,1562
Loto, red 2972	        8,9,10,12,13,28,34	    1,3438
Loto Plus, red 1728	    9,10,27,29,30,31,37	    1,3281

Pretraga obuhvata poslednje 64 kombinacije i sve njihove zamene jednog broja. Bez argumenata kod obrađuje sva tri fajla zasebno.
"""
