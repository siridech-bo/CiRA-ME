<template>
  <div class="bmd">
    <svg viewBox="0 0 820 360" width="100%" style="max-width: 780px; height: auto; display: block;"
      role="img" aria-label="Diagram of a bearing-mounted electric motor with accelerometers on the drive-end and fan-end housings">
      <!-- ===== foundation ===== -->
      <line x1="90" y1="312" x2="620" y2="312" class="ln strong" />
      <g class="ln">
        <path d="M120 312 l-14 22 M150 312 l14 22" />
        <path d="M470 312 l-14 22 M500 312 l14 22" />
      </g>
      <!-- feet -->
      <rect x="200" y="240" width="26" height="72" class="fill-body ln" rx="2" />
      <rect x="414" y="240" width="26" height="72" class="fill-body ln" rx="2" />

      <!-- ===== motor housing ===== -->
      <rect x="170" y="120" width="300" height="120" rx="14" class="fill-body ln" />
      <!-- cooling fins -->
      <g class="ln thin">
        <line v-for="x in fins" :key="x" :x1="x" y1="110" :x2="x" y2="120" />
      </g>
      <!-- terminal (junction) box -->
      <rect x="296" y="96" width="60" height="24" rx="3" class="fill-body2 ln" />
      <text x="326" y="112" class="lbl tiny mid">terminal box</text>

      <!-- fan cowl (fan-end / non-drive-end) -->
      <path d="M170 132 L138 146 L138 214 L170 228 Z" class="fill-body2 ln" />
      <g class="ln thin">
        <line x1="146" y1="150" x2="146" y2="210" />
        <line x1="154" y1="152" x2="154" y2="208" />
        <line x1="162" y1="154" x2="162" y2="206" />
      </g>

      <!-- drive-end bracket + bearing housing -->
      <rect x="462" y="146" width="20" height="68" rx="3" class="fill-body2 ln" />
      <circle cx="472" cy="180" r="16" class="fill-body ln" />
      <circle cx="472" cy="180" r="5" class="ln" fill="none" />

      <!-- shaft -->
      <rect x="482" y="173" width="96" height="14" rx="3" class="fill-shaft" />
      <!-- coupling -->
      <rect x="560" y="164" width="26" height="32" rx="3" class="fill-body2 ln" />
      <g class="ln thin"><line x1="566" y1="164" x2="566" y2="196" /><line x1="573" y1="164" x2="573" y2="196" /><line x1="580" y1="164" x2="580" y2="196" /></g>

      <!-- driven load -->
      <rect x="586" y="140" width="150" height="80" rx="10" class="fill-body ln" />
      <text x="661" y="176" class="lbl mid">Driven load</text>
      <text x="661" y="194" class="lbl tiny mid">pump · fan · gearbox</text>

      <!-- rotation arrow around shaft -->
      <path d="M520 150 a22 22 0 1 1 -14 -6" class="arw-rot" fill="none" />
      <path d="M506 144 l1 12 l11 -5 z" class="arw-rot-head" />
      <text x="524" y="138" class="lbl accent-rot">ω = {{ shaftHz }} Hz</text>

      <!-- ===== accelerometers ===== -->
      <!-- Drive-end accelerometer -->
      <g>
        <rect x="452" y="56" width="40" height="30" rx="4" class="fill-de" />
        <line x1="472" y1="86" x2="472" y2="162" class="arw-meas" />
        <path d="M472 168 l-6 -12 l12 0 z" class="arw-meas-head" />
        <text x="472" y="46" class="lbl mid strongtxt">Drive-end accelerometer</text>
        <text x="472" y="75" class="lbl tiny mid ontop">accel_de</text>
      </g>
      <!-- Fan-end accelerometer -->
      <g>
        <rect x="196" y="56" width="40" height="30" rx="4" class="fill-fe" />
        <line x1="216" y1="86" x2="216" y2="116" class="arw-meas" />
        <path d="M216 122 l-6 -12 l12 0 z" class="arw-meas-head" />
        <text x="216" y="46" class="lbl mid strongtxt">Fan-end accelerometer</text>
        <text x="216" y="75" class="lbl tiny mid ontop">accel_fe</text>
      </g>

      <!-- bearing detail callout -->
      <g transform="translate(70,250)">
        <circle cx="0" cy="0" r="34" class="fill-body ln" />
        <circle cx="0" cy="0" r="13" class="ln" fill="none" />
        <g class="fill-ball">
          <circle v-for="(b, i) in balls" :key="i" :cx="b.x" :cy="b.y" r="4.5" />
        </g>
        <text x="0" y="52" class="lbl tiny mid">{{ nBalls }} rolling elements</text>
        <text x="0" y="-44" class="lbl tiny mid accent-fault">BPFO · BPFI · BSF · FTF</text>
      </g>
    </svg>

    <p class="cap text-caption">
      Mount the accelerometer on the <strong>bearing housing</strong>, oriented radially (pointing at the shaft),
      as close to the bearing load zone as possible. This demo uses two real sensors — the
      <span class="k de">drive-end</span> and <span class="k fe">fan-end</span> accelerometers
      (<code>accel_de</code>, <code>accel_fe</code>). The drive-end usually sees a fault strongest.
    </p>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'

const props = withDefaults(defineProps<{ shaftHz?: number | string; nBalls?: number | string }>(), {
  shaftHz: 30, nBalls: 8,
})

const fins = [190, 205, 220, 235, 250, 265, 280, 340, 355, 370, 385, 400, 415, 430, 445]

// Ball positions evenly spaced around the bearing callout circle (radius ~23.5).
const balls = computed(() => {
  const n = Math.max(3, Math.min(24, Number(props.nBalls) || 8))
  const r = 23.5
  return Array.from({ length: n }, (_, i) => {
    const a = (2 * Math.PI * i) / n - Math.PI / 2
    return { x: +(r * Math.cos(a)).toFixed(2), y: +(r * Math.sin(a)).toFixed(2) }
  })
})
</script>

<style scoped>
.bmd { width: 100%; }
/* Structural line work follows the theme's text colour. */
.ln { stroke: currentColor; stroke-opacity: 0.55; fill: none; stroke-width: 1.6; }
.ln.strong { stroke-opacity: 0.75; stroke-width: 2; }
.ln.thin { stroke-width: 1; stroke-opacity: 0.4; }
.fill-body { fill: rgba(127, 127, 127, 0.08); }
.fill-body2 { fill: rgba(127, 127, 127, 0.16); }
.fill-shaft { fill: #90a4ae; }
.fill-ball { fill: #ffa726; }
.fill-de { fill: #42a5f5; }
.fill-fe { fill: #66bb6a; }

.lbl { fill: currentColor; fill-opacity: 0.85; font-size: 13px; font-family: inherit; }
.lbl.tiny { font-size: 11px; fill-opacity: 0.7; }
.lbl.mid { text-anchor: middle; }
.lbl.strongtxt { fill-opacity: 0.95; font-weight: 600; }
.lbl.ontop { fill: #fff; fill-opacity: 0.95; font-weight: 600; }
.accent-rot { fill: #66bb6a; fill-opacity: 0.95; }
.accent-meas { fill: #66bb6a; fill-opacity: 0.9; }
.accent-fault { fill: #ffa726; fill-opacity: 0.95; }

.arw-rot { stroke: #66bb6a; stroke-width: 2; }
.arw-rot-head { fill: #66bb6a; }
.arw-meas { stroke: #66bb6a; stroke-width: 2; stroke-dasharray: 4 3; }
.arw-meas-head { fill: #66bb6a; }

.cap { margin-top: 6px; opacity: 0.85; line-height: 1.5; }
.cap code { font-size: 0.85em; padding: 0 3px; border-radius: 3px; background: rgba(127, 127, 127, 0.16); }
.k { font-weight: 600; }
.k.de { color: #42a5f5; }
.k.fe { color: #66bb6a; }
</style>
