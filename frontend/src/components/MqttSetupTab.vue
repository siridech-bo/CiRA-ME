<template>
  <div class="mqtt-setup-tab">
    <!-- Loading / not-configured states -->
    <v-card v-if="!config" variant="tonal" color="grey-lighten-3">
      <v-card-text class="text-center py-8">
        <v-icon size="48" color="grey">mdi-file-tree-outline</v-icon>
        <p class="text-body-1 mt-3 mb-1">Asset tree not yet configured.</p>
        <p class="text-body-2 text-medium-emphasis">
          Run the first-run wizard to create a tree — sensor setup needs
          it to know the topic format.
        </p>
      </v-card-text>
    </v-card>

    <template v-else>
      <p class="text-body-2 text-medium-emphasis mb-4">
        Point any MQTT sensor at CiRA ME with three settings. Everything
        below reflects <strong>your</strong> current tree config — no
        manual editing needed.
      </p>

      <!-- ── ① Topic format ─────────────────────────────────────────── -->
      <v-card variant="tonal" class="mb-4">
        <v-card-title class="d-flex align-center py-2">
          <v-avatar color="primary" size="24" class="mr-2">
            <span class="text-caption font-weight-bold">1</span>
          </v-avatar>
          <span class="text-subtitle-1">Topic format</span>
          <v-spacer />
          <v-chip size="x-small" variant="tonal">
            {{ levelCount }} segments
          </v-chip>
        </v-card-title>
        <v-card-text class="pt-0">
          <CopyBlock :text="topicPattern" />
          <p class="text-caption text-medium-emphasis mt-2">
            Your tree levels: <code>{{ levelNames.join(' / ') }}</code>.
            Each segment must match a real node in the tree
            (Strict mode) or will be auto-created (Learn mode).
          </p>

          <div v-if="exampleMachine" class="mt-3">
            <p class="text-caption mb-1">
              <strong>Example</strong> —
              publishing a reading for
              <code>{{ exampleMachine.name }}</code>:
            </p>
            <CopyBlock :text="exampleTopic" />
          </div>
        </v-card-text>
      </v-card>

      <!-- ── ② Payload format ────────────────────────────────────────── -->
      <v-card variant="tonal" class="mb-4">
        <v-card-title class="d-flex align-center py-2">
          <v-avatar color="primary" size="24" class="mr-2">
            <span class="text-caption font-weight-bold">2</span>
          </v-avatar>
          <span class="text-subtitle-1">Payload — any of these three formats</span>
        </v-card-title>
        <v-card-text class="pt-0">
          <CopyBlock :text="'{&quot;value&quot;: 5.2}'" label="Recommended" />
          <CopyBlock :text="'{&quot;v&quot;: 5.2}'" />
          <CopyBlock :text="'5.2'" label="Bare number" />
          <v-alert
            type="info"
            variant="tonal"
            density="compact"
            class="mt-2"
          >
            <strong>Don't include a timestamp</strong> — the router adds
            one automatically when it writes the CSV.
          </v-alert>
        </v-card-text>
      </v-card>

      <!-- ── ③ Broker connection ─────────────────────────────────────── -->
      <v-card variant="tonal" class="mb-4">
        <v-card-title class="d-flex align-center py-2">
          <v-avatar color="primary" size="24" class="mr-2">
            <span class="text-caption font-weight-bold">3</span>
          </v-avatar>
          <span class="text-subtitle-1">Broker connection</span>
        </v-card-title>
        <v-card-text class="pt-0">
          <v-row dense>
            <v-col cols="12" md="6">
              <p class="text-caption text-medium-emphasis mb-1">Host</p>
              <CopyBlock :text="brokerHost" />
              <p class="text-caption text-medium-emphasis mt-2">
                Use <code>cirame-mosquitto</code> from other Docker
                containers on the same network, or the host machine's
                IP from a device on the LAN.
              </p>
            </v-col>
            <v-col cols="12" md="3">
              <p class="text-caption text-medium-emphasis mb-1">Port</p>
              <CopyBlock text="1883" />
            </v-col>
            <v-col cols="12" md="3">
              <p class="text-caption text-medium-emphasis mb-1">Auth</p>
              <div class="broker-value">none</div>
            </v-col>
          </v-row>
        </v-card-text>
      </v-card>

      <v-divider class="my-6" />

      <!-- ── Test-right-now snippets ─────────────────────────────────── -->
      <h3 class="text-subtitle-1 font-weight-bold mb-3">
        <v-icon size="18" class="mr-1">mdi-play-circle-outline</v-icon>
        Test right now
      </h3>

      <v-expansion-panels variant="accordion" multiple v-model="openSnippets">
        <v-expansion-panel>
          <v-expansion-panel-title>
            <v-icon start size="18">mdi-console</v-icon>
            Terminal (mosquitto_pub)
          </v-expansion-panel-title>
          <v-expansion-panel-text>
            <CopyBlock :text="mosquittoPubSnippet" multiline />
            <p class="text-caption text-medium-emphasis mt-2">
              Install: <code>apt install mosquitto-clients</code> (Linux)
              or <code>brew install mosquitto</code> (macOS).
            </p>
          </v-expansion-panel-text>
        </v-expansion-panel>

        <v-expansion-panel>
          <v-expansion-panel-title>
            <v-icon start size="18">mdi-language-python</v-icon>
            Python (paho-mqtt)
          </v-expansion-panel-title>
          <v-expansion-panel-text>
            <CopyBlock :text="pythonSnippet" multiline />
            <p class="text-caption text-medium-emphasis mt-2">
              Install: <code>pip install paho-mqtt</code>.
            </p>
          </v-expansion-panel-text>
        </v-expansion-panel>

        <v-expansion-panel>
          <v-expansion-panel-title>
            <v-icon start size="18">mdi-chip</v-icon>
            Arduino / ESP32 (PubSubClient)
          </v-expansion-panel-title>
          <v-expansion-panel-text>
            <CopyBlock :text="arduinoSnippet" multiline />
            <p class="text-caption text-medium-emphasis mt-2">
              Library: <code>PubSubClient</code> by Nick O'Leary.
              Call from your <code>loop()</code> at your sample rate.
            </p>
          </v-expansion-panel-text>
        </v-expansion-panel>

        <v-expansion-panel>
          <v-expansion-panel-title>
            <v-icon start size="18">mdi-memory</v-icon>
            Arduino UNO Q (Python side)
          </v-expansion-panel-title>
          <v-expansion-panel-text>
            <CopyBlock :text="unoQPythonSnippet" multiline />
            <p class="text-caption text-medium-emphasis mt-2">
              The UNO Q's Linux side ships with Python 3 —
              <code>pip install paho-mqtt</code>. Use the App Lab
              <code>brick</code> / <code>msg</code> bridge to pull
              readings from the MCU side (sensors on Modulino / I²C /
              analog), then publish to CiRA ME from Python. A
              <code>systemd</code> unit keeps it running after reboot.
            </p>
          </v-expansion-panel-text>
        </v-expansion-panel>

        <v-expansion-panel>
          <v-expansion-panel-title>
            <v-icon start size="18">mdi-nodejs</v-icon>
            Node.js (mqtt package)
          </v-expansion-panel-title>
          <v-expansion-panel-text>
            <CopyBlock :text="nodeSnippet" multiline />
            <p class="text-caption text-medium-emphasis mt-2">
              Install: <code>npm install mqtt</code>.
            </p>
          </v-expansion-panel-text>
        </v-expansion-panel>
      </v-expansion-panels>

      <v-divider class="my-6" />

      <!-- ── Next-steps helpers ──────────────────────────────────────── -->
      <v-row>
        <v-col cols="12" md="6">
          <v-card
            variant="tonal"
            color="success"
            :to="{ name: 'simulators' }"
            hover
          >
            <v-card-text class="d-flex align-center">
              <v-icon size="32" class="mr-3">mdi-gauge</v-icon>
              <div>
                <p class="text-body-2 font-weight-bold mb-0">
                  Don't have a real sensor yet?
                </p>
                <p class="text-caption mb-0">
                  Machine Simulators publishes correctly-formatted
                  messages against your tree.
                </p>
              </div>
            </v-card-text>
          </v-card>
        </v-col>
        <v-col cols="12" md="6">
          <v-card
            variant="tonal"
            color="warning"
            hover
            @click="$emit('go-to-rejected')"
          >
            <v-card-text class="d-flex align-center">
              <v-icon size="32" class="mr-3">mdi-cancel</v-icon>
              <div>
                <p class="text-body-2 font-weight-bold mb-0">
                  Sensor publishing but nothing appearing?
                </p>
                <p class="text-caption mb-0">
                  Check the Rejected tab — every rejection has a reason.
                </p>
              </div>
            </v-card-text>
          </v-card>
        </v-col>
      </v-row>
    </template>
  </div>
</template>

<script setup lang="ts">
/**
 * MQTT Sensor Setup guide — landing tab of the MQTT Rules page.
 *
 * Everything on this page is derived from the tree config passed in
 * via `config` prop, so a customer sees their actual topic pattern
 * and their actual registered machines — not a generic template.
 */
import { computed, ref } from 'vue'
import { useAssetTreeStore } from '@/stores/assetTree'
import CopyBlock from './CopyBlock.vue'

interface ConfigProp {
  level_names?: string[]
  root_name?: string
  topic_mode?: string
}

const props = defineProps<{
  config: ConfigProp | null
}>()

defineEmits<{
  (e: 'go-to-rejected'): void
}>()

const assetTreeStore = useAssetTreeStore()

const openSnippets = ref<number[]>([0])  // terminal snippet expanded by default

const levelNames = computed(() =>
  props.config?.level_names || ['factory', 'plant', 'machine', 'sensor'],
)
const levelCount = computed(() => levelNames.value.length)
const rootName = computed(() => props.config?.root_name || 'factory')

// Topic template: <root>/<L2>/<L3>/.../<sensor_leaf>
const topicPattern = computed(() => {
  const parts = [rootName.value]
  for (let i = 1; i < levelNames.value.length; i++) {
    parts.push(`<${levelNames.value[i]}>`)
  }
  return parts.join('/')
})

// Pick the first ACTIVE machine + first sensor for the example. Walks
// the store tree; falls back to a placeholder when nothing's registered
// yet (e.g. brand-new install right after wizard).
const exampleMachine = computed(() => {
  const tree = assetTreeStore.tree
  if (!tree || tree.length === 0) return null
  const machineLevel = levelCount.value - 2 // sensors are the leaf
  const findMachine = (nodes: any[]): { path: string; name: string; sensors: string[] } | null => {
    for (const n of nodes) {
      if (n.status !== 'active') continue
      if (n.level === machineLevel) {
        const sensors = (n.children || [])
          .filter((c: any) => c.status === 'active')
          .map((c: any) => c.name)
        return {
          name: n.name,
          path: n.topic_path,
          sensors,
        }
      }
      if (n.children?.length) {
        const found = findMachine(n.children)
        if (found) return found
      }
    }
    return null
  }
  return findMachine(tree)
})

const exampleSensor = computed(() =>
  exampleMachine.value?.sensors[0] || 'pressure',
)
const exampleTopic = computed(() =>
  exampleMachine.value
    ? `${exampleMachine.value.path}/${exampleSensor.value}`
    : `${topicPattern.value.replace(/<[^>]+>/g, 'demo')}`,
)

// Broker host: default to service name; user can copy their LAN IP
// separately.
const brokerHost = computed(() => 'cirame-mosquitto')

// ── Snippets ────────────────────────────────────────────────────────
const mosquittoPubSnippet = computed(() =>
  `mosquitto_pub -h <broker-host> -p 1883 \\
  -t "${exampleTopic.value}" \\
  -m '{"value": 5.2}'`,
)

const pythonSnippet = computed(() =>
  `import paho.mqtt.publish as pub
import json

pub.single(
    topic="${exampleTopic.value}",
    payload=json.dumps({"value": 5.2}),
    hostname="<broker-host>",
    port=1883,
)`,
)

const arduinoSnippet = computed(() =>
  `#include <PubSubClient.h>
#include <WiFi.h>

WiFiClient wifi;
PubSubClient client(wifi);

void setup() {
  WiFi.begin("<ssid>", "<password>");
  client.setServer("<broker-host>", 1883);
  client.connect("compressor_01");
}

void loop() {
  float pressure = readSensor();
  char payload[32];
  snprintf(payload, sizeof(payload), "{\\"value\\": %.2f}", pressure);
  client.publish("${exampleTopic.value}", payload);
  delay(1000);  // 1 Hz
}`,
)

const unoQPythonSnippet = computed(() =>
  `# On the Arduino UNO Q Linux side. paho-mqtt handles the network;
# the App Lab bridge pulls sensor readings from the MCU (Modulino
# breakout, I2C, or analog pins connected to the classic Arduino side).

import time, json
import paho.mqtt.publish as pub
from arduino.app_bricks.modulino import Pressure   # example brick

BROKER  = "<broker-host>"
TOPIC   = "${exampleTopic.value}"
SAMPLE_HZ = 1.0

sensor = Pressure()   # reads over I2C via the UNO Q bridge

while True:
    reading = sensor.read()          # e.g. 5.2 bar
    pub.single(
        topic=TOPIC,
        payload=json.dumps({"value": reading}),
        hostname=BROKER,
        port=1883,
    )
    time.sleep(1.0 / SAMPLE_HZ)

# Deploy as a service so it starts on boot:
#   sudo cp cira_publisher.service /etc/systemd/system/
#   sudo systemctl enable --now cira_publisher`,
)

const nodeSnippet = computed(() =>
  `import mqtt from 'mqtt'

const client = mqtt.connect('mqtt://<broker-host>:1883')

client.on('connect', () => {
  setInterval(() => {
    const value = readSensor()
    client.publish(
      '${exampleTopic.value}',
      JSON.stringify({ value }),
    )
  }, 1000)  // 1 Hz
})`,
)
</script>

<style scoped>
.mqtt-setup-tab code {
  background: rgba(var(--v-theme-on-surface), 0.08);
  padding: 1px 5px;
  border-radius: 3px;
  font-size: 0.85em;
}
.broker-value {
  font-family: 'JetBrains Mono', 'Fira Code', ui-monospace, monospace;
  padding: 8px 12px;
  background: rgba(var(--v-theme-on-surface), 0.05);
  border-radius: 4px;
  font-size: 0.9rem;
}
</style>
