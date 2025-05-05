/// <reference types="cypress" />

describe('Wiki-QA happy path', () => {
  beforeEach(() => {
    cy.visit('/')
  })

  it('shows pipeline all green & answer bubble', () => {
    // Type question and submit
    cy.get('input[placeholder="Ask MasterChat…"]').type('Who is Alan Turing?{enter}')
    
    // Wait for answer bubble
    cy.get('.history').should('contain', 'Alan Turing')
    
    // Wait for health overlay to disappear (all services healthy)
    cy.get('.overlay').should('not.exist')
    
    // Check debug logs for no errors
    cy.get('.terminal pre').should('not.contain', '[ERR]')
  })

  it('handles service errors gracefully', () => {
    // Wait for health check
    cy.get('.overlay').should('exist')
    
    // Check debug logs for health status
    cy.get('.terminal pre').should('contain', '[ERR]')
  })
}) 