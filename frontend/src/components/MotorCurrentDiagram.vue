<template>
  <div class="mcd">
    <svg viewBox="0 0 820 330" width="100%" style="max-width: 780px; height: auto; display: block;"
      role="img" aria-label="Diagram of an induction motor with clamp-on current transducers on the three supply phases">
      <!-- foundation -->
      <line x1="300" y1="286" x2="600" y2="286" class="ln strong" />
      <g class="ln"><path d="M340 286 l-14 20 M368 286 l14 20" /><path d="M520 286 l-14 20 M548 286 l14 20" /></g>
      <rect x="372" y="230" width="26" height="56" class="fill-body ln" rx="2" />
      <rect x="524" y="230" width="26" height="56" class="fill-body ln" rx="2" />

      <!-- supply panel / drive -->
      <rect x="58" y="120" width="72" height="118" rx="6" class="fill-body2 ln" />
      <text x="94" y="176" class="lbl mid">Supply</text>
      <text x="94" y="194" class="lbl mid">panel /</text>
      <text x="94" y="212" class="lbl mid">drive</text>
      <text x="94" y="110" class="lbl tiny mid accent-rot">f&#8320; = {{ lineHz }} Hz</text>

      <!-- three supply phases -->
      <g class="cable"><line x1="130" y1="150" x2="330" y2="150" /><line x1="130" y1="178" x2="330" y2="178" /><line x1="130" y1="206" x2="330" y2="206" /></g>

      <!-- clamp-on CTs -->
      <text x="225" y="130" class="lbl tiny mid">clamp-on current sensors (CTs)</text>
      <g>
        <circle cx="225" cy="150" r="12" class="clamp ph-a" /><path d="M219 141 a12 12 0 0 1 12 0" class="clamp-jaw" />
        <text x="246" y="154" class="lbl small ph-a-txt">Ia</text>
      </g>
      <g>
        <circle cx="225" cy="178" r="12" class="clamp ph-b" /><path d="M219 169 a12 12 0 0 1 12 0" class="clamp-jaw" />
        <text x="246" y="182" class="lbl small ph-b-txt">Ib</text>
      </g>
      <g>
        <circle cx="225" cy="206" r="12" class="clamp ph-c" /><path d="M219 197 a12 12 0 0 1 12 0" class="clamp-jaw" />
        <text x="246" y="210" class="lbl small ph-c-txt">Ic</text>
      </g>

      <!-- motor terminal box + body -->
      <rect x="316" y="150" width="22" height="60" rx="3" class="fill-body2 ln" />
      <rect x="338" y="110" width="250" height="120" rx="14" class="fill-body ln" />
      <g class="ln thin"><line v-for="x in fins" :key="x" :x1="x" y1="100" :x2="x" y2="110" /></g>
      <text x="463" y="175" class="lbl mid">induction motor</text>

      <!-- shaft + load -->
      <rect x="588" y="163" width="74" height="14" rx="3" class="fill-shaft" />
      <rect x="662" y="128" width="120" height="84" rx="10" class="fill-body ln" />
      <text x="722" y="166" class="lbl mid">motor load</text>
      <text x="722" y="184" class="lbl tiny mid">pump · fan · compressor</text>

      <!-- rotation -->
      <path d="M636 150 a20 20 0 1 1 -13 -5" class="arw-rot" fill="none" />
      <path d="M623 145 l1 11 l10 -5 z" class="arw-rot-head" />
      <text x="612" y="140" class="lbl tiny accent-rot">ω, slip s</text>
    </svg>

    <p class="cap text-caption">
      MCSA clamps a current transducer around each of the three stator supply phases —
      <span class="k a">Ia</span>, <span class="k b">Ib</span>, <span class="k c">Ic</span> —
      with <strong>no contact</strong> to the rotating machine. Broken rotor bars, eccentricity and
      load faults appear as sidebands around the line frequency <code>f&#8320;</code>.
    </p>
  </div>
</template>

<script setup lang="ts">
withDefaults(defineProps<{ lineHz?: number | string }>(), { lineHz: 50 })
const fins = [356, 371, 386, 401, 416, 431, 446, 461, 476, 491, 506, 521, 536, 551, 566]
</script>

<style scoped>
.mcd { width: 100%; }
.ln { stroke: currentColor; stroke-opacity: 0.55; fill: none; stroke-width: 1.6; }
.ln.strong { stroke-opacity: 0.75; stroke-width: 2; }
.ln.thin { stroke-width: 1; stroke-opacity: 0.4; }
.fill-body { fill: rgba(127, 127, 127, 0.08); }
.fill-body2 { fill: rgba(127, 127, 127, 0.16); }
.fill-shaft { fill: #90a4ae; }
.cable line { stroke: currentColor; stroke-opacity: 0.6; stroke-width: 2.4; }

.clamp { fill: none; stroke-width: 3; }
.clamp-jaw { fill: none; stroke: currentColor; stroke-opacity: 0.5; stroke-width: 2; }
.ph-a { stroke: #42a5f5; } .ph-b { stroke: #66bb6a; } .ph-c { stroke: #ffa726; }

.lbl { fill: currentColor; fill-opacity: 0.85; font-size: 13px; font-family: inherit; }
.lbl.tiny { font-size: 11px; fill-opacity: 0.7; }
.lbl.small { font-size: 12px; font-weight: 600; }
.lbl.mid { text-anchor: middle; }
.ph-a-txt { fill: #42a5f5; } .ph-b-txt { fill: #66bb6a; } .ph-c-txt { fill: #ffa726; }
.accent-rot { fill: #66bb6a; fill-opacity: 0.95; }
.arw-rot { stroke: #66bb6a; stroke-width: 2; } .arw-rot-head { fill: #66bb6a; }

.cap { margin-top: 6px; opacity: 0.85; line-height: 1.5; }
.cap code { font-size: 0.9em; padding: 0 3px; border-radius: 3px; background: rgba(127, 127, 127, 0.16); }
.k { font-weight: 600; }
.k.a { color: #42a5f5; } .k.b { color: #66bb6a; } .k.c { color: #ffa726; }
</style>
