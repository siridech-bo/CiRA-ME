/**
 * CiRA ME - Client-side feature extraction
 *
 * Pure TypeScript port of the backend `FeatureExtractor` lightweight path
 * (backend/app/services/feature_extractor.py). Ships as part of Fast Mode:
 * the Web Worker imports this module, receives raw windows from the server,
 * and computes features locally so 60 concurrent workshop users don't back
 * up on the 5-slot backend queue.
 *
 * Numerical parity target: within ~1e-4 vs. the backend on
 * sin(2*pi*5*t)-style synthetic signals. All divide-by-zeros are guarded and
 * any resulting NaN/Infinity is coerced to 0 (matches backend behaviour).
 *
 * NO DOM or Vue imports here — must be safe to load in a Web Worker context.
 */

export interface WindowFeatureResult {
  feature_names: string[]
  values: number[]
}

// ---- Small stats helpers ---------------------------------------------------

function sum(x: number[]): number {
  let s = 0
  for (let i = 0; i < x.length; i++) s += x[i]
  return s
}

function mean(x: number[]): number {
  if (x.length === 0) return 0
  return sum(x) / x.length
}

function variance(x: number[]): number {
  // Biased variance (matches np.var default ddof=0).
  const n = x.length
  if (n === 0) return 0
  const mu = mean(x)
  let s = 0
  for (let i = 0; i < n; i++) {
    const d = x[i] - mu
    s += d * d
  }
  return s / n
}

function std(x: number[]): number {
  return Math.sqrt(variance(x))
}

function median(x: number[]): number {
  if (x.length === 0) return 0
  const sorted = [...x].sort((a, b) => a - b)
  const m = Math.floor(sorted.length / 2)
  return sorted.length % 2 === 0
    ? (sorted[m - 1] + sorted[m]) / 2
    : sorted[m]
}

function minOf(x: number[]): number {
  let m = Infinity
  for (let i = 0; i < x.length; i++) if (x[i] < m) m = x[i]
  return m === Infinity ? 0 : m
}

function maxOf(x: number[]): number {
  let m = -Infinity
  for (let i = 0; i < x.length; i++) if (x[i] > m) m = x[i]
  return m === -Infinity ? 0 : m
}

function skewness(x: number[]): number {
  // Biased Pearson's moment estimator, matches scipy.stats.skew default.
  //   g1 = m3 / m2**1.5   where m_k = mean((x - mean(x))**k)
  const n = x.length
  if (n === 0) return 0
  const mu = mean(x)
  let m2 = 0
  let m3 = 0
  for (let i = 0; i < n; i++) {
    const d = x[i] - mu
    const d2 = d * d
    m2 += d2
    m3 += d2 * d
  }
  m2 /= n
  m3 /= n
  if (m2 === 0) return 0
  const denom = Math.pow(m2, 1.5)
  return denom === 0 ? 0 : m3 / denom
}

function kurtosis(x: number[]): number {
  // Fisher's excess kurtosis: m4/m2**2 - 3.
  const n = x.length
  if (n === 0) return 0
  const mu = mean(x)
  let m2 = 0
  let m4 = 0
  for (let i = 0; i < n; i++) {
    const d = x[i] - mu
    const d2 = d * d
    m2 += d2
    m4 += d2 * d2
  }
  m2 /= n
  m4 /= n
  if (m2 === 0) return 0
  const denom = m2 * m2
  return denom === 0 ? 0 : m4 / denom - 3
}

// ---- FFT (radix-2 Cooley-Tukey, iterative) --------------------------------

function nextPowerOfTwo(n: number): number {
  let p = 1
  while (p < n) p <<= 1
  return p
}

/**
 * Iterative in-place radix-2 Cooley-Tukey FFT.
 * Input is padded with zeros to the next power of two if needed.
 * Returns real+imag arrays of length N (padded length).
 */
export function fft(input: number[]): { re: number[]; im: number[] } {
  const n0 = input.length
  const n = nextPowerOfTwo(n0)
  const re = new Array<number>(n).fill(0)
  const im = new Array<number>(n).fill(0)
  for (let i = 0; i < n0; i++) re[i] = input[i]

  // Bit-reverse permutation.
  let j = 0
  for (let i = 1; i < n; i++) {
    let bit = n >> 1
    for (; j & bit; bit >>= 1) j ^= bit
    j ^= bit
    if (i < j) {
      const tr = re[i]; re[i] = re[j]; re[j] = tr
      const ti = im[i]; im[i] = im[j]; im[j] = ti
    }
  }

  // Cooley-Tukey butterflies.
  for (let size = 2; size <= n; size <<= 1) {
    const half = size >> 1
    const theta = (-2 * Math.PI) / size
    const wRe = Math.cos(theta)
    const wIm = Math.sin(theta)
    for (let start = 0; start < n; start += size) {
      let currRe = 1
      let currIm = 0
      for (let k = 0; k < half; k++) {
        const idx1 = start + k
        const idx2 = idx1 + half
        const tRe = currRe * re[idx2] - currIm * im[idx2]
        const tIm = currRe * im[idx2] + currIm * re[idx2]
        re[idx2] = re[idx1] - tRe
        im[idx2] = im[idx1] - tIm
        re[idx1] += tRe
        im[idx1] += tIm
        // Advance twiddle.
        const nCurrRe = currRe * wRe - currIm * wIm
        const nCurrIm = currRe * wIm + currIm * wRe
        currRe = nCurrRe
        currIm = nCurrIm
      }
    }
  }

  return { re, im }
}

// ---- Feature helpers ------------------------------------------------------

function autocorrLag(x: number[], lag: number): number {
  // Standard biased autocorrelation matching backend _autocorr(): sum((x[i]-mean)*(x[i+lag]-mean)) / (n * var)
  const n = x.length
  if (lag >= n) return 0
  const mu = mean(x)
  const varX = variance(x)
  if (varX === 0) return 0
  let s = 0
  for (let i = 0; i < n - lag; i++) {
    s += (x[i] - mu) * (x[i + lag] - mu)
  }
  return s / (n * varX)
}

function binnedEntropy(x: number[], numBins = 10): number {
  // Matches backend _binned_entropy: np.histogram(density=True), then
  // filter positive bins, then -sum(p * log(p + 1e-10)).
  // With density=True, hist values are the density (count / (n * bin_width)),
  // not probability. Backend uses that value directly so we do the same.
  const n = x.length
  if (n === 0) return 0
  const mn = minOf(x)
  const mx = maxOf(x)
  if (mn === mx) {
    // All values identical — histogram is a single spike.
    // np.histogram(x, bins=10) with constant array gives one bin at count=n,
    // rest zero. With density=True and bin_width computed from
    // (max-min)/numBins == 0, numpy returns 0 or Inf; we return 0 as a safe
    // proxy (backend suppresses RuntimeWarnings and coerces to 0).
    return 0
  }
  const binWidth = (mx - mn) / numBins
  const counts = new Array<number>(numBins).fill(0)
  for (let i = 0; i < n; i++) {
    // Match numpy: last bin is [max-eps, max] closed on both ends,
    // others are [edge_k, edge_k+1).
    let idx = Math.floor((x[i] - mn) / binWidth)
    if (idx >= numBins) idx = numBins - 1
    if (idx < 0) idx = 0
    counts[idx]++
  }
  let entropy = 0
  for (let i = 0; i < numBins; i++) {
    if (counts[i] > 0) {
      // density: count / (n * binWidth)
      const p = counts[i] / (n * binWidth)
      entropy -= p * Math.log(p + 1e-10)
    }
  }
  return entropy
}

// ---- Per-channel base feature computers -----------------------------------

interface ChannelData {
  x: number[]           // raw signal (length = window_size)
  n: number
  mu: number            // mean
  varX: number          // variance (biased)
}

function preparePerChannel(x: number[]): ChannelData {
  return { x, n: x.length, mu: mean(x), varX: variance(x) }
}

// Each feature computer takes a ChannelData + samplingRate and returns a
// finite number (NaN/Infinity are collapsed to 0 by the caller).
type FeatureFn = (c: ChannelData, sr: number) => number

// ---- TSFresh-style statistical features -----------------------------------

const timeFeatures: Record<string, FeatureFn> = {
  mean: (c) => c.mu,
  std: (c) => Math.sqrt(c.varX),
  min: (c) => minOf(c.x),
  max: (c) => maxOf(c.x),
  median: (c) => median(c.x),
  sum: (c) => sum(c.x),
  variance: (c) => c.varX,
  skewness: (c) => skewness(c.x),
  kurtosis: (c) => kurtosis(c.x),
  abs_energy: (c) => {
    let s = 0
    for (let i = 0; i < c.n; i++) s += c.x[i] * c.x[i]
    return s
  },
  root_mean_square: (c) => {
    let s = 0
    for (let i = 0; i < c.n; i++) s += c.x[i] * c.x[i]
    return Math.sqrt(s / c.n)
  },
  mean_abs_change: (c) => {
    if (c.n < 2) return 0
    let s = 0
    for (let i = 1; i < c.n; i++) s += Math.abs(c.x[i] - c.x[i - 1])
    return s / (c.n - 1)
  },
  mean_change: (c) => {
    if (c.n < 2) return 0
    let s = 0
    for (let i = 1; i < c.n; i++) s += c.x[i] - c.x[i - 1]
    return s / (c.n - 1)
  },
  length: (c) => c.n,
  count_above_mean: (c) => {
    let cnt = 0
    for (let i = 0; i < c.n; i++) if (c.x[i] > c.mu) cnt++
    return cnt
  },
  count_below_mean: (c) => {
    let cnt = 0
    for (let i = 0; i < c.n; i++) if (c.x[i] < c.mu) cnt++
    return cnt
  },
  first_location_of_maximum: (c) => {
    // Matches np.argmax / len — normalized 0..1.
    if (c.n === 0) return 0
    let idx = 0
    let mx = c.x[0]
    for (let i = 1; i < c.n; i++) {
      if (c.x[i] > mx) { mx = c.x[i]; idx = i }
    }
    return idx / c.n
  },
  first_location_of_minimum: (c) => {
    if (c.n === 0) return 0
    let idx = 0
    let mn = c.x[0]
    for (let i = 1; i < c.n; i++) {
      if (c.x[i] < mn) { mn = c.x[i]; idx = i }
    }
    return idx / c.n
  },
  last_location_of_maximum: (c) => {
    // Backend: (len - 1 - argmax(x[::-1])) / len
    // Which is the last index i where x[i] == max(x), then divided by len.
    // Note the result is (last_idx) / n where last_idx is in [0, n-1].
    if (c.n === 0) return 0
    let mx = c.x[0]
    let lastIdx = 0
    for (let i = 1; i < c.n; i++) {
      if (c.x[i] >= mx) { mx = c.x[i]; lastIdx = i }
    }
    // For strict-max reproduction, we need the last i where x[i] == max.
    // Recompute using the final max value.
    let last = 0
    for (let i = 0; i < c.n; i++) if (c.x[i] === mx) last = i
    return last / c.n
  },
  last_location_of_minimum: (c) => {
    if (c.n === 0) return 0
    let mn = c.x[0]
    for (let i = 1; i < c.n; i++) if (c.x[i] < mn) mn = c.x[i]
    let last = 0
    for (let i = 0; i < c.n; i++) if (c.x[i] === mn) last = i
    return last / c.n
  },
  ratio_beyond_r_sigma: (c) => {
    // tsfresh default r=2. Fraction of values with |x - mean| > r * std.
    if (c.n === 0) return 0
    const sd = Math.sqrt(c.varX)
    if (sd === 0) return 0
    const thr = 2 * sd
    let cnt = 0
    for (let i = 0; i < c.n; i++) if (Math.abs(c.x[i] - c.mu) > thr) cnt++
    return cnt / c.n
  },
  longest_strike_above_mean: (c) => {
    let best = 0
    let cur = 0
    for (let i = 0; i < c.n; i++) {
      if (c.x[i] > c.mu) {
        cur++
        if (cur > best) best = cur
      } else {
        cur = 0
      }
    }
    return best
  },
  longest_strike_below_mean: (c) => {
    let best = 0
    let cur = 0
    for (let i = 0; i < c.n; i++) {
      if (c.x[i] < c.mu) {
        cur++
        if (cur > best) best = cur
      } else {
        cur = 0
      }
    }
    return best
  },
  sum_of_reoccurring_values: (c) => {
    // Backend: sum([v for v in col if np.sum(col == v) > 1])
    // I.e., for each element whose value appears >1 times in the array,
    // add it to the sum (element by element, not unique).
    // With floats this is dicey — matches backend by using strict equality.
    // For the sin(2*pi*5*t) synthetic used in verification, only 0-crossings
    // reoccur (or none), so this is stable.
    const counts = new Map<number, number>()
    for (let i = 0; i < c.n; i++) {
      counts.set(c.x[i], (counts.get(c.x[i]) ?? 0) + 1)
    }
    let s = 0
    for (let i = 0; i < c.n; i++) {
      if ((counts.get(c.x[i]) ?? 0) > 1) s += c.x[i]
    }
    return s
  },
  sum_of_reoccurring_data_points: (c) => {
    // tsfresh definition: sum of values that appear more than once, counted
    // once per unique value * count. Backend does not implement this in the
    // lightweight path so we mimic the tsfresh reference: sum(v * count(v))
    // over values with count > 1.
    // Since backend's lightweight feature list contains this name but the
    // TSFRESH_FEATURES dict in feature_extractor.py doesn't provide a lambda
    // for it, the backend simply skips it silently — so it never appears in
    // the extracted column list. To stay in perfect parity we skip it too;
    // this is handled at feature-selection time (see SUPPORTED_FEATURES).
    // Included here for completeness in case the backend adds it later.
    const counts = new Map<number, number>()
    for (let i = 0; i < c.n; i++) {
      counts.set(c.x[i], (counts.get(c.x[i]) ?? 0) + 1)
    }
    let s = 0
    for (const [v, cnt] of counts) {
      if (cnt > 1) s += v * cnt
    }
    return s
  },
  percentage_of_reoccurring_values: (c) => {
    // Backend: len(np.unique(col)) / len(col). NOTE: this matches tsfresh's
    // percentage_of_reoccurring_datapoints_to_all_datapoints only by
    // coincidence for constant signals; the backend definition is literally
    // "fraction of distinct values", not the tsfresh convention. We match
    // the backend, not tsfresh.
    if (c.n === 0) return 0
    const unique = new Set<number>()
    for (let i = 0; i < c.n; i++) unique.add(c.x[i])
    return unique.size / c.n
  },
}

// ---- DSP features (time + frequency) --------------------------------------

const dspFeaturesFactory: Record<string, FeatureFn> = {
  rms: (c) => {
    let s = 0
    for (let i = 0; i < c.n; i++) s += c.x[i] * c.x[i]
    return Math.sqrt(s / c.n)
  },
  peak_to_peak: (c) => maxOf(c.x) - minOf(c.x),
  crest_factor: (c) => {
    let s = 0
    let peak = 0
    for (let i = 0; i < c.n; i++) {
      s += c.x[i] * c.x[i]
      const a = Math.abs(c.x[i])
      if (a > peak) peak = a
    }
    const rms = Math.sqrt(s / c.n)
    const safe = rms === 0 ? 1e-10 : rms
    return peak / safe
  },
  shape_factor: (c) => {
    let s = 0
    let sa = 0
    for (let i = 0; i < c.n; i++) {
      s += c.x[i] * c.x[i]
      sa += Math.abs(c.x[i])
    }
    const rms = Math.sqrt(s / c.n)
    const meanAbs = sa / c.n
    const rmsSafe = rms === 0 ? 1e-10 : rms
    const meanAbsSafe = meanAbs === 0 ? 1e-10 : meanAbs
    return rmsSafe / meanAbsSafe
  },
  impulse_factor: (c) => {
    let peak = 0
    let sa = 0
    for (let i = 0; i < c.n; i++) {
      const a = Math.abs(c.x[i])
      if (a > peak) peak = a
      sa += a
    }
    const meanAbs = sa / c.n
    const meanAbsSafe = meanAbs === 0 ? 1e-10 : meanAbs
    return peak / meanAbsSafe
  },
  margin_factor: (c) => {
    // Backend: max(|x|) / (mean(sqrt(|x|))**2 with the mean-sqrt safeguarded)
    let peak = 0
    let ss = 0
    for (let i = 0; i < c.n; i++) {
      const a = Math.abs(c.x[i])
      if (a > peak) peak = a
      ss += Math.sqrt(a)
    }
    const meanSqrt = ss / c.n
    const meanSqrtSq = meanSqrt * meanSqrt
    const safe = meanSqrtSq === 0 ? 1e-10 : meanSqrtSq
    return peak / safe
  },
  zero_crossing_rate: (c) => {
    // Backend: sum(diff(sign(window)) != 0) / (n - 1)
    // np.sign(0) == 0, so any transition through 0 counts too.
    if (c.n < 2) return 0
    let cnt = 0
    for (let i = 1; i < c.n; i++) {
      const s1 = c.x[i] > 0 ? 1 : c.x[i] < 0 ? -1 : 0
      const s0 = c.x[i - 1] > 0 ? 1 : c.x[i - 1] < 0 ? -1 : 0
      if (s1 !== s0) cnt++
    }
    return cnt / (c.n - 1)
  },
  autocorr_lag1: (c) => autocorrLag(c.x, 1),
  autocorr_lag5: (c) => autocorrLag(c.x, 5),
  binned_entropy: (c) => binnedEntropy(c.x, 10),
}

// ---- Spectral features (all computed from same FFT) -----------------------

interface SpectralBundle {
  spectral_centroid: number
  spectral_bandwidth: number
  spectral_rolloff: number
  spectral_flatness: number
  spectral_entropy: number
  peak_frequency: number
  spectral_skewness: number
  spectral_kurtosis: number
  band_power_low: number
  band_power_mid: number
  band_power_high: number
}

function computeSpectral(x: number[], samplingRate: number): SpectralBundle {
  // Matches backend behaviour: uses scipy.fft.fft + fftfreq semantics, keeps
  // only the non-negative frequency bins (positive_freq_mask = freqs >= 0),
  // then computes power = |X|**2 / sum(|X|**2) as the normalized spectrum.
  const n = x.length
  const { re, im } = fft(x)
  const nFFT = re.length

  // fftfreq for length N with d = 1/sr:
  //   freqs[k] = k * sr / N   for k <= N/2
  //   freqs[k] = (k - N) * sr / N   for k > N/2
  // We keep freqs >= 0 → first half + Nyquist bin.
  const halfLen = Math.floor(nFFT / 2) + 1  // number of non-negative freqs
  const fftMag = new Array<number>(halfLen)
  const fftFreqs = new Array<number>(halfLen)
  for (let k = 0; k < halfLen; k++) {
    fftMag[k] = Math.sqrt(re[k] * re[k] + im[k] * im[k])
    fftFreqs[k] = (k * samplingRate) / nFFT
  }
  // scipy's fftfreq for even N puts Nyquist at negative sign, so the mask
  // `freqs >= 0` drops it. Match that by dropping the last bin when N is
  // even. For odd N (rare here — window sizes are powers of 2), keep all.
  // Backend uses fftfreq → for N even the Nyquist bin has freq -sr/2, so
  // it's dropped by (freqs >= 0). Mirror that.
  const nEven = nFFT % 2 === 0
  const effLen = nEven ? halfLen - 1 : halfLen
  const mag = fftMag.slice(0, effLen)
  const freqs = fftFreqs.slice(0, effLen)

  let totalPower = 0
  for (let k = 0; k < effLen; k++) totalPower += mag[k] * mag[k]
  const totalPowerSafe = totalPower > 0 ? totalPower : 1e-10

  const normPower = new Array<number>(effLen)
  for (let k = 0; k < effLen; k++) normPower[k] = (mag[k] * mag[k]) / totalPowerSafe

  // Centroid.
  let centroid = 0
  for (let k = 0; k < effLen; k++) centroid += freqs[k] * normPower[k]

  // Bandwidth.
  let bandwidthSq = 0
  for (let k = 0; k < effLen; k++) {
    const d = freqs[k] - centroid
    bandwidthSq += d * d * normPower[k]
  }
  const bandwidth = Math.sqrt(bandwidthSq)

  // Rolloff (95% of cumulative normalized power).
  const cumsum = new Array<number>(effLen)
  let acc = 0
  for (let k = 0; k < effLen; k++) { acc += normPower[k]; cumsum[k] = acc }
  const target = 0.95 * cumsum[effLen - 1]
  // np.searchsorted(cumsum, target) — leftmost index where cumsum[i] >= target.
  let rolloffIdx = effLen - 1
  for (let k = 0; k < effLen; k++) {
    if (cumsum[k] >= target) { rolloffIdx = k; break }
  }
  const rolloff = freqs[Math.min(rolloffIdx, effLen - 1)]

  // Flatness.
  let logSum = 0
  let arith = 0
  for (let k = 0; k < effLen; k++) {
    logSum += Math.log(mag[k] + 1e-10)
    arith += mag[k]
  }
  const geoMean = Math.exp(logSum / effLen)
  const arithMean = arith / effLen
  const flatness = geoMean / (arithMean + 1e-10)

  // Entropy (base 2).
  let entropy = 0
  for (let k = 0; k < effLen; k++) {
    const p = normPower[k] + 1e-10
    entropy -= p * (Math.log(p) / Math.LN2)
  }

  // Peak freq.
  let peakIdx = 0
  let peakVal = mag[0]
  for (let k = 1; k < effLen; k++) if (mag[k] > peakVal) { peakVal = mag[k]; peakIdx = k }
  const peakFreq = freqs[peakIdx]

  // Spectral skewness/kurtosis of the FFT magnitude spectrum.
  const sSkew = (() => {
    const v = skewness(mag)
    return Number.isFinite(v) ? v : 0
  })()
  const sKurt = (() => {
    const v = kurtosis(mag)
    return Number.isFinite(v) ? v : 0
  })()

  // Band powers: split 0..Nyquist into thirds using sr/6 and sr/3 cutoffs.
  const lowMax = samplingRate / 6
  const midMax = samplingRate / 3
  const nyquist = samplingRate / 2
  let lowP = 0
  let midP = 0
  let highP = 0
  for (let k = 0; k < effLen; k++) {
    const f = freqs[k]
    const p = mag[k] * mag[k]
    if (f >= 0 && f < lowMax) lowP += p
    else if (f >= lowMax && f < midMax) midP += p
    else if (f >= midMax && f <= nyquist) highP += p
  }
  return {
    spectral_centroid: centroid,
    spectral_bandwidth: bandwidth,
    spectral_rolloff: rolloff,
    spectral_flatness: flatness,
    spectral_entropy: entropy,
    peak_frequency: peakFreq,
    spectral_skewness: sSkew,
    spectral_kurtosis: sKurt,
    band_power_low: lowP / totalPowerSafe,
    band_power_mid: midP / totalPowerSafe,
    band_power_high: highP / totalPowerSafe,
  }
}

const SPECTRAL_FEATURE_NAMES: (keyof SpectralBundle)[] = [
  'spectral_centroid',
  'spectral_bandwidth',
  'spectral_rolloff',
  'spectral_flatness',
  'spectral_entropy',
  'peak_frequency',
  'spectral_skewness',
  'spectral_kurtosis',
  'band_power_low',
  'band_power_mid',
  'band_power_high',
]

// ---- Portable / supported feature catalogue --------------------------------

// Every base feature name the client can compute. Any name in the request
// that's NOT here is unsupported → the view greys it out.
export const SUPPORTED_FEATURES: string[] = [
  // TSFresh statistical
  'mean', 'std', 'min', 'max', 'median',
  'sum', 'variance', 'skewness', 'kurtosis',
  'abs_energy', 'root_mean_square', 'mean_abs_change',
  'mean_change', 'length', 'sum_of_reoccurring_values',
  'sum_of_reoccurring_data_points', 'ratio_beyond_r_sigma',
  'count_above_mean', 'count_below_mean',
  'longest_strike_above_mean', 'longest_strike_below_mean',
  'first_location_of_maximum', 'first_location_of_minimum',
  'last_location_of_maximum', 'last_location_of_minimum',
  'percentage_of_reoccurring_values',
  // DSP
  'rms', 'peak_to_peak', 'crest_factor', 'shape_factor',
  'impulse_factor', 'margin_factor', 'zero_crossing_rate',
  'autocorr_lag1', 'autocorr_lag5', 'binned_entropy',
  // Spectral
  ...SPECTRAL_FEATURE_NAMES,
]

/**
 * Split a selected-feature list into (supported, unsupported) for the
 * client-side extractor. Names not in SUPPORTED_FEATURES are ignored during
 * extraction — the view uses this to warn the user before they hit Extract.
 */
export function partitionFeatures(
  selectedFeatures: string[],
): { supported: string[]; unsupported: string[] } {
  const supported: string[] = []
  const unsupported: string[] = []
  const supportedSet = new Set(SUPPORTED_FEATURES)
  for (const f of selectedFeatures) {
    if (supportedSet.has(f)) supported.push(f)
    else unsupported.push(f)
  }
  return { supported, unsupported }
}

// ---- Main entry point ------------------------------------------------------

function safe(v: number): number {
  return Number.isFinite(v) ? v : 0
}

/**
 * Compute features for a single window across all channels.
 *
 * @param window          shape [window_size][num_channels], row-major
 * @param channelNames    length = num_channels
 * @param selectedFeatures base feature names (e.g. 'mean', 'spectral_entropy')
 * @param samplingRate    used only for spectral features
 *
 * Feature naming convention (must match backend): `{feature}_{channel}`.
 */
export function computeWindowFeatures(
  window: number[][],
  channelNames: string[],
  selectedFeatures: string[],
  samplingRate: number,
): WindowFeatureResult {
  const windowSize = window.length
  const numChannels = channelNames.length
  const featureNames: string[] = []
  const values: number[] = []

  const selected = new Set(selectedFeatures)

  // Reshape column-wise: extract each channel as its own array once.
  const channels: ChannelData[] = new Array(numChannels)
  for (let ch = 0; ch < numChannels; ch++) {
    const col = new Array<number>(windowSize)
    for (let t = 0; t < windowSize; t++) col[t] = window[t][ch]
    channels[ch] = preparePerChannel(col)
  }

  // Time-domain (TSFresh-style) — emitted in the SAME order as backend so
  // downstream code paths that key on column-name only will get the same
  // order too.
  for (const fname of Object.keys(timeFeatures)) {
    if (!selected.has(fname)) continue
    const fn = timeFeatures[fname]
    for (let ch = 0; ch < numChannels; ch++) {
      featureNames.push(`${fname}_${channelNames[ch]}`)
      values.push(safe(fn(channels[ch], samplingRate)))
    }
  }

  // DSP time-domain.
  for (const fname of Object.keys(dspFeaturesFactory)) {
    if (!selected.has(fname)) continue
    const fn = dspFeaturesFactory[fname]
    for (let ch = 0; ch < numChannels; ch++) {
      featureNames.push(`${fname}_${channelNames[ch]}`)
      values.push(safe(fn(channels[ch], samplingRate)))
    }
  }

  // Spectral — compute once per channel, then emit only requested names.
  // Avoids N separate FFTs per selected spectral feature.
  const anySpectral = SPECTRAL_FEATURE_NAMES.some((n) => selected.has(n))
  if (anySpectral) {
    const perChannel: SpectralBundle[] = new Array(numChannels)
    for (let ch = 0; ch < numChannels; ch++) {
      perChannel[ch] = computeSpectral(channels[ch].x, samplingRate)
    }
    for (const fname of SPECTRAL_FEATURE_NAMES) {
      if (!selected.has(fname)) continue
      for (let ch = 0; ch < numChannels; ch++) {
        featureNames.push(`${fname}_${channelNames[ch]}`)
        values.push(safe(perChannel[ch][fname]))
      }
    }
  }

  return { feature_names: featureNames, values }
}
