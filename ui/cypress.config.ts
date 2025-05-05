import { defineConfig } from 'cypress'
import { exec } from 'node:child_process'
import { promisify } from 'node:util'

const execAsync = promisify(exec)

export default defineConfig({
  projectId: '8p7jhd',
  e2e: {
    baseUrl: process.env.CYPRESS_BASE_URL || 'http://localhost:5173',
    supportFile: 'cypress/support/e2e.ts',
    specPattern: 'cypress/e2e/**/*.cy.{js,jsx,ts,tsx}',
    video: true,
    screenshotOnRunFailure: true,
    defaultCommandTimeout: 10000,
    pageLoadTimeout: 30000,
    requestTimeout: 10000,
    responseTimeout: 30000,
    setupNodeEvents(on, config) {
      on('task', {
        async execCommand(command: string) {
          try {
            if (command.includes('docker compose')) {
              await execAsync('cd .. && ' + command)
            } else {
              await execAsync(command)
            }
            return null
          } catch (error) {
            console.error(`Failed to execute command: ${command}`, error)
            throw error
          }
        }
      })
    }
  },
  component: {
    devServer: {
      framework: 'vue',
      bundler: 'vite'
    },
    supportFile: 'cypress/support/component.ts',
    specPattern: 'cypress/component/**/*.cy.{js,jsx,ts,tsx}'
  },
  chromeWebSecurity: false,
  viewportWidth: 1280,
  viewportHeight: 800,
  retries: {
    runMode: 2,
    openMode: 0
  }
}) 