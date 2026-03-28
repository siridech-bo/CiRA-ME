/**
 * CiRA ME - MQTT Composable
 * Connects to MQTT broker via WebSocket for live sensor data streaming.
 */

import { ref, shallowRef, onUnmounted } from 'vue'
import mqtt from 'mqtt'
import type { MqttClient, IClientOptions } from 'mqtt'

export interface MqttState {
  client: ReturnType<typeof shallowRef<MqttClient | null>>
  connected: ReturnType<typeof ref<boolean>>
  error: ReturnType<typeof ref<string | null>>
  messageCount: ReturnType<typeof ref<number>>
  messagesPerSec: ReturnType<typeof ref<number>>
}

export function useMqtt() {
  const client = shallowRef<MqttClient | null>(null)
  const connected = ref(false)
  const error = ref<string | null>(null)
  const messageCount = ref(0)
  const messagesPerSec = ref(0)

  // Rate counter
  let rateCounter = 0
  let rateInterval: ReturnType<typeof setInterval> | null = null

  function connect(brokerUrl: string, options: Partial<IClientOptions> = {}) {
    // Disconnect existing
    if (client.value) {
      disconnect()
    }

    error.value = null

    try {
      const c = mqtt.connect(brokerUrl, {
        clientId: `cira-${Date.now()}-${Math.random().toString(16).slice(2, 8)}`,
        clean: true,
        keepalive: 30,
        reconnectPeriod: 3000,
        connectTimeout: 10000,
        ...options,
      })

      c.on('connect', () => {
        connected.value = true
        error.value = null
      })

      c.on('close', () => {
        connected.value = false
      })

      c.on('error', (err: Error) => {
        error.value = err.message
        connected.value = false
      })

      c.on('reconnect', () => {
        error.value = null
      })

      c.on('offline', () => {
        connected.value = false
      })

      client.value = c

      // Start rate counter
      rateInterval = setInterval(() => {
        messagesPerSec.value = rateCounter
        rateCounter = 0
      }, 1000)

      return c
    } catch (e: any) {
      error.value = e.message || 'Failed to connect'
      return null
    }
  }

  function subscribe(topic: string, qos: 0 | 1 | 2 = 0) {
    if (!client.value) return
    client.value.subscribe(topic, { qos })
  }

  function onMessage(callback: (topic: string, payload: any) => void) {
    if (!client.value) return
    client.value.on('message', (topic: string, payload: Buffer) => {
      messageCount.value++
      rateCounter++
      try {
        const data = JSON.parse(payload.toString())
        callback(topic, data)
      } catch {
        // Binary or non-JSON payload — pass raw
        callback(topic, payload)
      }
    })
  }

  function disconnect() {
    if (rateInterval) {
      clearInterval(rateInterval)
      rateInterval = null
    }
    if (client.value) {
      client.value.end(true)
      client.value = null
    }
    connected.value = false
    messageCount.value = 0
    messagesPerSec.value = 0
  }

  onUnmounted(disconnect)

  return {
    client,
    connected,
    error,
    messageCount,
    messagesPerSec,
    connect,
    subscribe,
    onMessage,
    disconnect,
  }
}
