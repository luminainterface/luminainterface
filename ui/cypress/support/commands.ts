/// <reference types="cypress" />

// Custom command for visiting the graph page with reliable loading
Cypress.Commands.add('visitGraph', () => {
  const base = Cypress.env('CYPRESS_BASE_URL') || 'http://localhost:5173'
  cy.visit(`${base}/graph`)
  cy.get('svg', { timeout: 10000 }).should('be.visible')
})

// Custom command for visiting the main page and opening metrics
Cypress.Commands.add('openMetricsPanel', () => {
  const base = Cypress.env('CYPRESS_BASE_URL') || 'http://localhost:5173'
  cy.visit(base)
  cy.get('button').contains('Metrics').click()
  cy.get('.metrics-panel', { timeout: 5000 }).should('be.visible')
  // Wait for charts to be ready
  cy.get('canvas').should('be.visible')
  cy.wait(1000) // Give charts time to animate
})

// Custom command to wait for API health
Cypress.Commands.add('waitForApiHealth', () => {
  cy.get('.overlay').should('not.exist', { timeout: 10000 })
})

declare global {
  namespace Cypress {
    interface Chainable {
      visitGraph(): Chainable<void>
      openMetricsPanel(): Chainable<void>
      waitForApiHealth(): Chainable<void>
    }
  }
} 