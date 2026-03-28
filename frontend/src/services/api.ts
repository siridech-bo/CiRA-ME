/**
 * CiRA ME - API Service
 * Axios instance with interceptors
 */

import axios from 'axios'

const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL || '',
  withCredentials: true,
  headers: {
    'Content-Type': 'application/json'
  }
})

// Request interceptor
api.interceptors.request.use(
  (config) => {
    // Add any request transformations here
    return config
  },
  (error) => {
    return Promise.reject(error)
  }
)

// Response interceptor
api.interceptors.response.use(
  (response) => {
    return response
  },
  (error) => {
    // Handle 401 Unauthorized — redirect to login
    // BUT not on public pages (standalone apps, published apps)
    if (error.response?.status === 401) {
      const path = window.location.pathname
      const isPublicPage = path.startsWith('/standalone/') || path.startsWith('/apps/')
      if (!isPublicPage && path !== '/login') {
        window.location.href = '/login'
      }
    }

    return Promise.reject(error)
  }
)

export default api
