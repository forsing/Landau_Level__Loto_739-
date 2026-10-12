#!/usr/bin/env python3

"""LANDAU LEVELS V2b: structural spectrum and level-occupation prediction."""

import argparse
import csv
from pathlib import Path
import numpy as np



DEFAULT = "/data/loto7_4698_k80.csv"
# DEFAULT = "/data/loto7_4698_k80_loto_2971.csv"
# DEFAULT = "/data/loto7_4698_k80_loto_plus_1727.csv"



N, K, START, WINDOW, LANDMARKS, LEVELS = 39, 7, 32, 768, 24, 12


def load(path):
    rows = []
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        for line, row in enumerate(csv.reader(stream), 1):
            if not row or all(not value.strip() for value in row):
                continue
            try:
                values = [int(value.strip()) for value in row]
            except ValueError as error:
                raise ValueError(
                    f"Red {line}: ocekujem sedam celih brojeva."
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
        np.zeros(len(rows)),
        rows,
        np.full(len(rows), N + 1),
    ))
    return np.column_stack((
        rows / (N + 1),
        (np.diff(edges, axis=1) - 1) / (N - K + 1),
    ))


def distances(a, b):
    return np.abs(
        a[:, None, :] - b[None, :, :]
    ).sum(axis=2)


class Spectrum:
    def __init__(self, history):
        layouts = shape(history)
        center = np.median(layouts, axis=0)
        first = int(np.argmin(
            np.abs(layouts - center).sum(axis=1)
        ))

        indices = [first]
        minimum = distances(
            layouts, layouts[[first]]
        )[:, 0]

        # Deterministicki izbor strukturnih oslonaca.
        while len(indices) < min(LANDMARKS, len(layouts)):
            chosen = int(np.argmax(minimum))
            if minimum[chosen] <= 1e-12:
                break
            indices.append(chosen)
            minimum = np.minimum(
                minimum,
                distances(layouts, layouts[[chosen]])[:, 0],
            )

        self.landmarks = layouts[indices]
        separation = distances(
            self.landmarks, self.landmarks
        )
        positive = separation[
            np.triu_indices(len(indices), 1)
        ]
        positive = positive[positive > 1e-12]
        self.bandwidth = (
            float(np.median(positive)) if len(positive) else 1.
        )

        # Eksponencijalna slicnost sa apsolutnim rastojanjem.
        kernel = np.exp(-separation / self.bandwidth)
        eigenvalues, eigenvectors = np.linalg.eigh(kernel)

        order = np.argsort(eigenvalues)[::-1]
        order = order[
            eigenvalues[order] > eigenvalues[order[0]] * 1e-10
        ][:LEVELS]

        eigenvalues = eigenvalues[order]
        self.basis = (
            eigenvectors[:, order]
            / np.sqrt(eigenvalues)[None, :]
        )

        # Efektivni strukturni energetski nivoi.
        self.energies = -np.log(
            eigenvalues / eigenvalues[0]
        )

    def occupations(self, rows):
        layouts = shape(rows)
        pieces = []

        for start in range(0, len(layouts), 2048):
            kernel = np.exp(
                -distances(
                    layouts[start:start+2048],
                    self.landmarks,
                ) / self.bandwidth
            )
            amplitudes = kernel @ self.basis
            squared = amplitudes * amplitudes
            norm = squared.sum(axis=1, keepdims=True)

            if np.any(norm <= 0) or not np.all(np.isfinite(norm)):
                raise ArithmeticError(
                    "Neispravan spektralni opis."
                )
            pieces.append(squared / norm)

        return np.vstack(pieces)


def fields(layouts, occupations, t):
    return np.r_[
        occupations[t-1],
        occupations[t-2],
        occupations[t-8:t].mean(axis=0),
        occupations[t-32:t].mean(axis=0),
        layouts[t-1],
        layouts[t-2],
    ]


def prepare(history):
    history = history[-WINDOW:]
    spectrum = Spectrum(history)
    layouts = shape(history)
    occupations = spectrum.occupations(history)

    features = np.asarray([
        fields(layouts, occupations, t)
        for t in range(START, len(history) + 1)
    ])

    return (
        spectrum,
        features[:-1],
        occupations[START:],
        features[-1],
        occupations.mean(axis=0),
    )


def simplex(values):
    # Nenegativne tezine sa zbirom jedan.
    ordered = np.sort(values)[::-1]
    cumulative = np.cumsum(ordered) - 1.
    count = np.arange(1, len(values) + 1)
    valid = np.flatnonzero(
        ordered - cumulative / count > 0
    )
    threshold = (
        cumulative[valid[-1]] / count[valid[-1]]
    )
    weights = np.maximum(values - threshold, 0)
    return weights / weights.sum()


def forecast(prepared, alpha):
    _, x, y, query, _ = prepared
    center = x.mean(axis=0)
    scale = np.maximum(x.std(axis=0), 1e-8)

    a = np.column_stack((
        np.ones(len(x)),
        (x-center) / scale,
    ))
    penalty = np.diag(np.r_[
        0., np.full(x.shape[1], alpha)
    ])
    coefficients = np.linalg.solve(
        a.T @ a + penalty,
        a.T @ y,
    )

    return simplex(
        np.r_[1., (query-center) / scale] @ coefficients
    )


def evaluate(rows, spectrum, weights):
    # Podudarnost predvidjene i kandidatove raspodele po nivoima.
    occupations = spectrum.occupations(rows)
    return np.sqrt(
        occupations * weights[None, :]
    ).sum(axis=1)**2


def seeds(history):
    candidates = [
        tuple(map(int, row)) for row in history[-32:]
    ]
    for step in range(1, 7):
        for first in range(1, N - (K-1)*step + 1):
            candidates.append(tuple(
                first + j*step for j in range(K)
            ))
    return np.asarray(sorted(set(candidates)), dtype=int)


def search(history, spectrum, weights):
    current = seeds(history)
    values = np.arange(1, N+1)

    for iteration in range(1, 101):
        blocks = [current]
        owners = [np.arange(len(current))]

        for j in range(K):
            lower = (
                current[:, j-1]+1
                if j else np.ones(len(current), dtype=int)
            )
            upper = (
                current[:, j+1]-1
                if j < K-1 else np.full(len(current), N)
            )

            who, which = np.nonzero(
                (values[None, :] >= lower[:, None])
                & (values[None, :] <= upper[:, None])
            )
            neighbors = current[who].copy()
            neighbors[:, j] = values[which]

            blocks.append(neighbors)
            owners.append(who)

        candidates = np.vstack(blocks)
        owners = np.concatenate(owners)
        scores = evaluate(candidates, spectrum, weights)

        maximum = np.full(len(current), -np.inf)
        np.maximum.at(maximum, owners, scores)

        indices = np.arange(len(candidates))
        eligible = scores >= maximum[owners] - 1e-12
        chosen = np.full(
            len(current), len(candidates), dtype=int
        )
        np.minimum.at(
            chosen,
            owners,
            np.where(eligible, indices, len(candidates)),
        )

        updated = candidates[chosen]
        if np.array_equal(updated, current):
            scores = evaluate(current, spectrum, weights)
            best = int(np.argmax(scores))
            return (
                tuple(map(int, current[best])),
                float(scores[best]),
                iteration,
            )
        current = updated

    raise RuntimeError(
        "Pretraga nije dostigla koordinatni optimum u 100 koraka."
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
        flush=True,
    )


def self_test():
    rows = np.asarray([
        [1, 2, 3, 4, 5, 6, 7],
        [4, 12, 18, 28, 35, 36, 37],
        [33, 34, 35, 36, 37, 38, 39],
        [2, 7, 12, 18, 24, 30, 39],
    ])
    history = np.tile(rows, (50, 1))
    prepared = prepare(history)
    spectrum = prepared[0]

    assert spectrum.energies[0] == 0
    assert np.all(
        np.diff(spectrum.energies) >= -1e-12
    )

    occupations = spectrum.occupations(history)
    assert np.all(occupations >= 0)
    assert np.allclose(occupations.sum(axis=1), 1)

    weights = forecast(prepared, 10.)
    assert np.all(weights >= 0)
    assert np.isclose(weights.sum(), 1)

    result = search(history, spectrum, weights)
    assert result == search(history, spectrum, weights)

    combo, score, _ = result
    assert len(set(combo)) == K
    assert list(combo) == sorted(combo)
    assert 0 <= score <= 1 + 1e-12

    for j in range(K):
        lo = combo[j-1]+1 if j else 1
        hi = combo[j+1]-1 if j < K-1 else N
        neighbors = np.tile(combo, (hi-lo+1, 1))
        neighbors[:, j] = np.arange(lo, hi+1)

        assert (
            evaluate(neighbors, spectrum, weights).max()
            <= score + 1e-12
        )

    print(
        "Self-test OK: spektar, tezine, ponovljivost "
        "i koordinatni optimum."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="?", default=DEFAULT)
    parser.add_argument("--validation", type=int, default=48)
    parser.add_argument("--test", type=int, default=64)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    rows = load(args.csv)
    if args.validation < 16 or args.test < 16:
        parser.error(
            "Validation i test moraju imati bar 16 kola."
        )

    split = len(rows) - args.test
    begin = split - args.validation
    if begin < START + 128:
        parser.error(
            "Premalo istorije za zadatu validaciju i test."
        )

    alphas = (1., 10., 100.)
    scores = [[] for _ in alphas]

    print(
        f"Ulaz: {Path(args.csv).resolve()}; "
        f"redova={len(rows)}",
        flush=True,
    )

    for t in range(begin, split):
        prepared = prepare(rows[:t])

        for j, alpha in enumerate(alphas):
            weights = forecast(prepared, alpha)
            combo, _, _ = search(
                rows[:t], prepared[0], weights
            )
            scores[j].append(hits(combo, rows[t]))

        if (t-begin+1) % 16 == 0:
            print(
                f"Validacija: {t-begin+1}/{args.validation}",
                flush=True,
            )

    means = [np.mean(values) for values in scores]
    best = int(np.argmax(means))
    alpha = alphas[best]

    print(
        f"Validacija: redovi {begin+1}..{split}; "
        f"alpha={alpha:g}; prosek={means[best]:.4f}",
        flush=True,
    )

    model_scores = []
    previous_scores = []
    unconditional_scores = []

    for t in range(split, len(rows)):
        prepared = prepare(rows[:t])
        weights = forecast(prepared, alpha)
        combo, _, _ = search(
            rows[:t], prepared[0], weights
        )

        model_scores.append(hits(combo, rows[t]))
        previous_scores.append(hits(rows[t-1], rows[t]))

        reference, _, _ = search(
            rows[:t], prepared[0], prepared[4]
        )
        unconditional_scores.append(hits(reference, rows[t]))

        if (t-split+1) % 16 == 0:
            print(
                f"Test: {t-split+1}/{args.test}",
                flush=True,
            )

    print(f"Test: redovi {split+1}..{len(rows)}")
    summarize("LANDAU LEVELS V2b", model_scores)
    summarize("Prethodna kombinacija", previous_scores)
    summarize(
        "Nivoi bez vremenskog uslovljavanja",
        unconditional_scores,
    )

    prepared = prepare(rows)
    weights = forecast(prepared, alpha)
    combo, score, iterations = search(
        rows, prepared[0], weights
    )

    print(
        f"Poslednji red: {','.join(map(str, rows[-1]))}"
    )
    print(
        f"SLEDECI RED {len(rows)+1}: "
        f"{','.join(map(str, combo))}"
    )
    print(
        f"Skor={score:.10f}; iteracija={iterations}"
    )

    for j, (energy, weight) in enumerate(zip(
        prepared[0].energies, weights
    )):
        print(
            f"Nivo {j}: E={energy:.6f}; "
            f"tezina={weight:.6f}"
        )

    print(
        "E je strukturni spektar. Skor rangira kombinacije; "
        "globalni maksimum pretrage nije potvrden."
    )


if __name__ == "__main__":
    main()



"""
Ulaz: /data/loto7_4698_k80.csv; redova=4698
Validacija: 16/48
Validacija: 32/48
Validacija: 48/48
Validacija: redovi 4587..4634; alpha=1; prosek=1.1667
Test: 16/64
Test: 32/64
Test: 48/64
Test: 64/64
Test: redovi 4635..4698
LANDAU LEVELS V2b: prosek=1.1562; pogodaka 0..7=[17, 24, 19, 4, 0, 0, 0, 0]
Prethodna kombinacija: prosek=1.2188; pogodaka 0..7=[13, 30, 16, 4, 1, 0, 0, 0]
Nivoi bez vremenskog uslovljavanja: prosek=1.3281; pogodaka 0..7=[16, 19, 22, 6, 1, 0, 0, 0]
Poslednji red: 4,12,18,28,35,36,37
SLEDECI RED 4699: 3,x,6,y,21,z,39
Skor=0.9957389345; iteracija=22
Nivo 0: E=-0.000000; tezina=0.701898
Nivo 1: E=1.558860; tezina=0.105450
Nivo 2: E=2.073303; tezina=0.043108
Nivo 3: E=2.206379; tezina=0.027000
Nivo 4: E=2.308761; tezina=0.022343
Nivo 5: E=2.402494; tezina=0.022718
Nivo 6: E=2.574118; tezina=0.025097
Nivo 7: E=2.632248; tezina=0.002892
Nivo 8: E=2.696498; tezina=0.029350
Nivo 9: E=2.737971; tezina=0.007240
Nivo 10: E=2.914790; tezina=0.005378
Nivo 11: E=2.972425; tezina=0.007525
E je strukturni spektar. Skor rangira kombinacije; globalni maksimum pretrage nije potvrden.



Ulaz: /data/loto7_4698_k80_loto_2971.csv; redova=2971
Validacija: 16/48
Validacija: 32/48
Validacija: 48/48
Validacija: redovi 2860..2907; alpha=1; prosek=1.4583
Test: 16/64
Test: 32/64
Test: 48/64
Test: 64/64
Test: redovi 2908..2971
LANDAU LEVELS V2b: prosek=1.2188; pogodaka 0..7=[18, 20, 20, 6, 0, 0, 0, 0]
Prethodna kombinacija: prosek=1.2969; pogodaka 0..7=[11, 30, 16, 7, 0, 0, 0, 0]
Nivoi bez vremenskog uslovljavanja: prosek=1.3594; pogodaka 0..7=[12, 20, 29, 3, 0, 0, 0, 0]
Poslednji red: 18,20,21,24,29,34,38
SLEDECI RED 2972: 5,x,17,y,31,z,35
Skor=0.9985165622; iteracija=31
Nivo 0: E=-0.000000; tezina=0.824533
Nivo 1: E=1.499461; tezina=0.060277
Nivo 2: E=1.907795; tezina=0.013836
Nivo 3: E=2.318078; tezina=0.034179
Nivo 4: E=2.364655; tezina=0.020472
Nivo 5: E=2.435694; tezina=0.016179
Nivo 6: E=2.509486; tezina=0.008119
Nivo 7: E=2.552198; tezina=0.006330
Nivo 8: E=2.697520; tezina=0.013611
Nivo 9: E=2.718540; tezina=0.000189
Nivo 10: E=2.922248; tezina=0.001163
Nivo 11: E=2.998157; tezina=0.001111
E je strukturni spektar. Skor rangira kombinacije; globalni maksimum pretrage nije potvrden.



Ulaz: /data/loto7_4698_k80_loto_plus_1727.csv; redova=1727
Validacija: 16/48
Validacija: 32/48
Validacija: 48/48
Validacija: redovi 1616..1663; alpha=10; prosek=1.2292
Test: 16/64
Test: 32/64
Test: 48/64
Test: 64/64
Test: redovi 1664..1727
LANDAU LEVELS V2b: prosek=1.3281; pogodaka 0..7=[14, 25, 16, 8, 1, 0, 0, 0]
Prethodna kombinacija: prosek=1.2969; pogodaka 0..7=[15, 23, 18, 8, 0, 0, 0, 0]
Nivoi bez vremenskog uslovljavanja: prosek=1.2500; pogodaka 0..7=[15, 25, 17, 7, 0, 0, 0, 0]
Poslednji red: 4,12,18,28,35,36,37
SLEDECI RED 1728: 5,x,11,y,21,z,39
Skor=0.9902051840; iteracija=14
Nivo 0: E=-0.000000; tezina=0.692592
Nivo 1: E=1.467715; tezina=0.090813
Nivo 2: E=1.980416; tezina=0.019401
Nivo 3: E=2.286667; tezina=0.040166
Nivo 4: E=2.334287; tezina=0.017108
Nivo 5: E=2.448007; tezina=0.029935
Nivo 6: E=2.473405; tezina=0.021139
Nivo 7: E=2.557958; tezina=0.030438
Nivo 8: E=2.638820; tezina=0.022745
Nivo 9: E=2.678840; tezina=0.014913
Nivo 10: E=2.853350; tezina=0.011032
Nivo 11: E=2.902174; tezina=0.009717
E je strukturni spektar. Skor rangira kombinacije; globalni maksimum pretrage nije potvrden.
"""



"""

QUANTUM THEORY & SOLID STATE PHYSICS - Landau levels, Landau diamagnetism, the theory of phase transitions

Landauovi nivoi imaju jasniji pravac za primenu.
V2b — LANDAU LEVELS. U fizici su to diskretni energetski nivoi. Ideja bi bila da strukture celih kombinacija opišem diskretnim strukturnim stanjima i učim prelaze između njih. Predikcija bi prvo izdvojila naredno stanje, zatim kombinaciju unutar njega. Ključno je da stanja proizlaze iz odnosa između položaja i razmaka, umesto iz proizvoljnog deljenja raspona na intervale. To je predlog analogije sa Landauovim nivoima.
V2c — LANDAU DIAMAGNETISM. Ovde bih istražio odziv distribucije na promenu uslova. U fizici se odziv dobija iz promene energije pod magnetnim poljem. Za moj model trebalo bi definisati merljivu promenu prethodnih konfiguracija i učiti kako se naredna struktura menja u odnosu na nju. Ne bih unapred zadavao da odgovor mora biti suprotan prethodnom pomeranju — smer i jačinu treba izvesti iz podataka. Inspiracija je Landauov rad o dijamagnetizmu.
V2c zahteva preciznije određenje šta u modelu predstavlja pobudu, a šta odziv.


Strukturne nivoe izvešću iz odnosa položaja i razmaka u csv fajlu, zatim proveriti predikcije kroz kola hronološki. 
Nivoe izvodi iz spektra strukturnih sličnosti, zatim uči njihovu raspodelu za naredno kolo.


Rezultati na poslednja 64 kola:

Metod	                            Prosečan broj pogodaka
V2b	                                1,1562
Prethodna kombinacija	            1,2188
Nivoi bez vremenskog uslovljavanja	1,3281


V2b na ovom testu nije pokazala prednost. Provereni su spektar, hronologija, normalizacija, ponovljivost i koordinatni optimum. Pretraga ne potvrđuje globalni maksimum.
"""
 
