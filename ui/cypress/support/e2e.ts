import '@cypress/code-coverage/support'
import '@testing-library/cypress/add-commands'
import type { exec } from 'child_process'
import './commands'
import { addMatchImageSnapshotCommand } from '@simonsmith/cypress-image-snapshot/command'

// Custom command to check debug logs
Cypress.Commands.add('checkDebugLogs', () => {
  cy.window().its('debugStore.logs').then((logs: any[]) => {
    const errors = logs.filter(log => log.level === 'error')
    expect(errors).to.have.length(0)
  })
})

// Custom command to wait for pipeline steps
Cypress.Commands.add('waitForPipelineStep', (step: string, status: string) => {
  cy.get('#pipelineViz').within(() => {
    cy.contains(step).should('have.class', `bg-${status === 'ok' ? 'green' : 'red'}-500`)
  })
})

// Custom command to wait for all pipeline steps to be green
Cypress.Commands.add('waitForPipelineGreen', () => {
  cy.get('#pipelineViz').within(() => {
    cy.contains('Crawl').should('have.class', 'bg-green-500')
    cy.contains('Summarise').should('have.class', 'bg-green-500')
    cy.contains('QA').should('have.class', 'bg-green-500')
  })
})

// Docker Compose commands
Cypress.Commands.add('dcUp', () => {
  cy.task('execCommand', 'docker compose up -d')
})

Cypress.Commands.add('dcDown', () => {
  cy.task('execCommand', 'docker compose down')
})

addMatchImageSnapshotCommand({
  failureThreshold: 0.03, // threshold for entire image
  failureThresholdType: 'percent', // percent of image or number of pixels
  customDiffConfig: { threshold: 0.1 }, // threshold for each pixel
  capture: 'viewport', // capture viewport in screenshot
})

declare global {
  namespace Cypress {
    interface Chainable {
      checkDebugLogs(): Chainable<void>
      waitForPipelineStep(step: string, status: string): Chainable<void>
      waitForPipelineGreen(): Chainable<void>
      dcUp(): Chainable<void>
      dcDown(): Chainable<void>
      matchImageSnapshot(name?: string, options?: any): Chainable<Element>
    }
  }
} 