// src/composables/useHands.ts
import { ref, onMounted, onUnmounted } from 'vue'

// Types for our hand tracking data
export interface HandLandmark {
  x: number
  y: number
  z: number
}

export interface Rotation {
  pitch: number
  yaw: number
}

export interface HandData {
  raw_landmarks: HandLandmark[]
  palm_center: [number, number, number]
  rotation: Rotation
  confidence: number
}

// Core composable for hand tracking
export function useHands(url: string = 'ws://localhost:8000/ws') {
  const isConnected = ref(false)
  const handData = ref<HandData | null>(null)
  const error = ref<string | null>(null)
  let socket: WebSocket | null = null
  let reconnectAttempts = 0
  const MAX_RECONNECT_ATTEMPTS = 5
  const RECONNECT_DELAY = 1000 // 1 second

  // Smoothed/processed values
  const palmPosition = ref<[number, number, number]>([0, 0, 0])
  const handRotation = ref<Rotation>({ pitch: 0, yaw: 0 })

  // Debug values
  const lastMessageTime = ref<number>(Date.now())
  const fps = ref<number>(0)
  let frameCount = 0
  let lastFpsUpdate = Date.now()

  const connect = () => {
    try {
      socket = new WebSocket(url)

      socket.onopen = () => {
        console.log('WebSocket connected')
        isConnected.value = true
        error.value = null
        reconnectAttempts = 0
      }

      socket.onmessage = (event) => {
        try {
          const now = Date.now()
          frameCount++

          // Update FPS every second
          if (now - lastFpsUpdate >= 1000) {
            fps.value = frameCount
            frameCount = 0
            lastFpsUpdate = now
          }

          const data: HandData = JSON.parse(event.data)
          handData.value = data
          lastMessageTime.value = now

          // Update smoothed values
          // Could add lerp/smoothing here if needed
          palmPosition.value = data.palm_center
          handRotation.value = data.rotation
        } catch (e) {
          console.error('Failed to parse hand tracking data:', e)
          error.value = 'Failed to parse hand tracking data'
        }
      }

      socket.onerror = (e) => {
        console.error('WebSocket error:', e)
        error.value = 'WebSocket error'
        isConnected.value = false
      }

      socket.onclose = () => {
        console.log('WebSocket closed')
        isConnected.value = false

        // Attempt reconnection if we haven't exceeded max attempts
        if (reconnectAttempts < MAX_RECONNECT_ATTEMPTS) {
          reconnectAttempts++
          console.log(`Attempting reconnection ${reconnectAttempts}/${MAX_RECONNECT_ATTEMPTS}...`)
          setTimeout(connect, RECONNECT_DELAY)
        } else {
          error.value = 'Maximum reconnection attempts reached'
        }
      }
    } catch (e) {
      console.error('Failed to connect:', e)
      error.value = 'Failed to connect to hand tracking server'
      isConnected.value = false
    }
  }

  const disconnect = () => {
    if (socket) {
      socket.close()
      socket = null
    }
  }

  // Utility functions
  const getLatency = () => {
    return Date.now() - lastMessageTime.value
  }

  const getFps = () => fps.value

  // Lifecycle hooks
  onMounted(() => {
    connect()
  })

  onUnmounted(() => {
    disconnect()
  })

  return {
    // Connection state
    isConnected,
    error,
    connect,
    disconnect,

    // Hand tracking data
    handData,
    palmPosition,
    handRotation,

    // Debug/metrics
    getLatency,
    getFps,
  }
}
