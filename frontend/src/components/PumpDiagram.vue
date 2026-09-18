<template>
  <div class="pmd">
    <svg viewBox="0 0 820 340" width="100%" style="max-width: 780px; height: auto; display: block;"
      role="img" :aria-label="showCurrent
        ? 'Diagram of a motor-driven centrifugal pump with an accelerometer on the pump bearing and clamp-on current sensors on the motor supply'
        : 'Diagram of a motor-driven centrifugal pump with an accelerometer on the pump bearing housing'">
      <!-- common baseplate -->
      <rect x="140" y="286" width="500" height="12" rx="2" class="fill-body2 ln" />
      <g class="ln"><path d="M170 298 l-12 16 M198 298 l12 16" /><path d="M560 298 l-12 16 M588 298 l12 16" /></g>

      <!-- ===== optional 3-phase supply + CTs (fusion) ===== -->
      <g v-if="showCurrent">
        <rect x="26" y="138" width="46" height="92" rx="5" class="fill-body2 ln" />
        <text x="49" y="126" class="lbl tiny mid">3~ supply</text>
        <g class="cable"><line x1="72" y1="150" x2="150" y2="150" /><line x1="72" y1="172" x2="150" y2="172" /><line x1="72" y1="194" x2="150" y2="194" /></g>
        <circle cx="108" cy="150" r="9" class="clamp ph-a" /><text x="92" y="154" class="lbl small ph-a-txt end">Ia</text>
        <circle cx="108" cy="172" r="9" class="clamp ph-b" /><text x="92" y="176" class="lbl small ph-b-txt end">Ib</text>
        <circle cx="108" cy="194" r="9" class="clamp ph-c" /><text x="92" y="198" class="lbl small ph-c-txt end">Ic</text>
        <rect x="138" y="150" width="14" height="46" rx="2" class="fill-body2 ln" />
      </g>

      <!-- ===== motor ===== -->
      <rect x="150" y="118" width="170" height="114" rx="14" class="fill-body ln" />
      <g class="ln thin"><line v-for="x in fins" :key="x" :x1="x" y1="108" :x2="x" y2="118" /></g>
      <text x="235" y="179" class="lbl mid">motor</text>
      <rect x="180" y="232" width="22" height="54" class="fill-body ln" /><rect x="288" y="232" width="22" height="54" class="fill-body ln" />

      <!-- coupling -->
      <rect x="320" y="162" width="26" height="28" rx="3" class="fill-body2 ln" />
      <g class="ln thin"><line x1="326" y1="162" x2="326" y2="190" /><line x1="333" y1="162" x2="333" y2="190" /><line x1="340" y1="162" x2="340" y2="190" /></g>

      <!-- pump bearing pedestal (accelerometer mount) -->
      <rect x="348" y="150" width="40" height="136" rx="3" class="fill-body ln" />
      <circle cx="368" cy="176" r="9" class="ln" fill="none" />
      <!-- shaft -->
      <rect x="346" y="169" width="166" height="12" rx="3" class="fill-shaft" />

      <!-- ===== centrifugal pump (volute) ===== -->
      <circle cx="512" cy="174" r="60" class="fill-body ln" />
      <circle cx="512" cy="174" r="46" class="ln thin" fill="none" />
      <g class="vane" :transform="`translate(512,174)`">
        <path v-for="(d, i) in vanes" :key="i" :d="d" />
      </g>
      <circle cx="512" cy="174" r="9" class="fill-shaft" />
      <!-- discharge (up) -->
      <rect x="500" y="70" width="24" height="46" rx="2" class="fill-body2 ln" />
      <text x="512" y="62" class="lbl tiny mid">discharge</text>
      <!-- suction (side) -->
      <rect x="570" y="162" width="52" height="24" rx="2" class="fill-body2 ln" />
      <text x="642" y="178" class="lbl tiny">suction</text>

      <!-- rotation -->
      <path d="M470 150 a18 18 0 1 1 -12 -5" class="arw-rot" fill="none" />
      <path d="M458 145 l1 10 l9 -4 z" class="arw-rot-head" />
      <text x="440" y="140" class="lbl tiny accent-rot">ω = {{ shaftHz }} Hz</text>

      <!-- ===== accelerometer on the pump bearing ===== -->
      <rect x="348" y="52" width="40" height="28" rx="4" class="fill-vib" />
      <line x1="368" y1="80" x2="368" y2="148" class="arw-meas" />
      <path d="M368 154 l-6 -12 l12 0 z" class="arw-meas-head" />
      <text x="368" y="42" class="lbl mid strongtxt">accelerometer</text>
      <text x="368" y="70" class="lbl tiny mid ontop">accel</text>
    </svg>

    <p class="cap text-caption">
      The <span class="k v">accelerometer</span> (<code>accel</code>) mounts radially on the pump
      bearing housing. Cavitation and impeller faults appear as blade-pass harmonics
      ({{ nBlades }} vanes × shaft speed) and broadband high-frequency energy.
      <template v-if="showCurrent">
        Fusion also clamps CTs on the three motor phases
        (<span class="k a">Ia</span>, <span class="k b">Ib</span>, <span class="k c">Ic</span>) —
        the Park-vector modulus adds a current view of the same fault, <strong>four channels</strong> in total.
      </template>
    </p>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'

const props = withDefaults(defineProps<{
  shaftHz?: number | string; nBlades?: number | string; showCurrent?: boolean; lineHz?: number | string
}>(), { shaftHz: 25, nBlades: 6, showCurrent: false, lineHz: 50 })

const fins = [170, 185, 200, 215, 230, 245, 260, 275, 290, 305]

// Backswept impeller vanes, drawn in the volute's local frame (centre at 0,0).
const vanes = computed(() => {
  const n = Math.max(3, Math.min(12, Number(props.nBlades) || 6))
  const rin = 12, rout = 44, sweep = 0.7
  return Array.from({ length: n }, (_, i) => {
    const a = (2 * Math.PI * i) / n
    const x1 = rin * Math.cos(a), y1 = rin * Math.sin(a)
    const x2 = rout * Math.cos(a + sweep), y2 = rout * Math.sin(a + sweep)
    const cx = rin * 1.9 * Math.cos(a + sweep * 0.4), cy = rin * 1.9 * Math.sin(a + sweep * 0.4)
    return `M${x1.toFixed(1)} ${y1.toFixed(1)} Q${cx.toFixed(1)} ${cy.toFixed(1)} ${x2.toFixed(1)} ${y2.toFixed(1)}`
  })
})
</script>

<style scoped>
.pmd { width: 100%; }
.ln { stroke: currentColor; stroke-opacity: 0.55; fill: none; stroke-width: 1.6; }
.ln.thin { stroke-width: 1; stroke-opacity: 0.4; }
.fill-body { fill: rgba(127, 127, 127, 0.08); }
.fill-body2 { fill: rgba(127, 127, 127, 0.16); }
.fill-shaft { fill: #90a4ae; }
.fill-vib { fill: #42a5f5; }
.cable line { stroke: currentColor; stroke-opacity: 0.6; stroke-width: 2.4; }
.vane path { stroke: currentColor; stroke-opacity: 0.5; stroke-width: 2; fill: none; }

.clamp { fill: none; stroke-width: 3; }
.ph-a { stroke: #42a5f5; } .ph-b { stroke: #66bb6a; } .ph-c { stroke: #ffa726; }

.lbl { fill: currentColor; fill-opacity: 0.85; font-size: 13px; font-family: inherit; }
.lbl.tiny { font-size: 11px; fill-opacity: 0.7; }
.lbl.small { font-size: 12px; font-weight: 600; }
.lbl.mid { text-anchor: middle; }
.lbl.end { text-anchor: end; }
.lbl.strongtxt { fill-opacity: 0.95; font-weight: 600; }
.lbl.ontop { fill: #fff; fill-opacity: 0.95; font-weight: 600; }
.ph-a-txt { fill: #42a5f5; } .ph-b-txt { fill: #66bb6a; } .ph-c-txt { fill: #ffa726; }
.accent-rot { fill: #66bb6a; fill-opacity: 0.95; }
.arw-rot { stroke: #66bb6a; stroke-width: 2; } .arw-rot-head { fill: #66bb6a; }
.arw-meas { stroke: #42a5f5; stroke-width: 2; stroke-dasharray: 4 3; }
.arw-meas-head { fill: #42a5f5; }

.cap { margin-top: 6px; opacity: 0.85; line-height: 1.5; }
.cap code { font-size: 0.9em; padding: 0 3px; border-radius: 3px; background: rgba(127, 127, 127, 0.16); }
.k { font-weight: 600; }
.k.v { color: #42a5f5; } .k.a { color: #42a5f5; } .k.b { color: #66bb6a; } .k.c { color: #ffa726; }
</style>
