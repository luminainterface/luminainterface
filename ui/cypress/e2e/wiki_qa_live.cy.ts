/// <reference types="cypress" />

describe('Wiki-QA Live Backend', () => {
  before(() => {
    cy.dcUp()
    // Set longer timeout for first-time model pulls
    Cypress.config('defaultCommandTimeout', 30000)
  })

  after(() => {
    cy.dcDown()
  })

  beforeEach(() => {
    cy.visit('/')
  })

  it('completes full pipeline with real backend', () => {
    // Type question and submit
    cy.get('input[placeholder="Ask MasterChat…"]').type('Who is Alan Turing?{enter}')

    // Wait for answer bubble
    cy.get('.history').should('contain', 'Alan Turing')

    // Verify health overlay is not visible (all services healthy)
    cy.get('.overlay').should('not.exist')

    // Check debug logs for no errors
    cy.get('.terminal pre').should('not.contain', '[ERR]')
  })

  it('handles service health correctly', () => {
    // Wait for health check
    cy.get('.overlay').should('exist')

    // Check debug logs for health status
    cy.get('.terminal pre').should('contain', '[ERR]')
  })
}) 