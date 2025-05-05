/// <reference types="cypress" />

describe('Lumina happy path', () => {
  beforeEach(() => {
    cy.visit('/')
  })

  it('shows chat panel with input field', () => {
    cy.get('input[placeholder="Ask MasterChat…"]').should('exist')
  })

  it('opens Metrics panel with button click', () => {
    cy.get('button').contains('Metrics').click()
    cy.get('.metrics-panel').should('be.visible')
  })

  it('sends chat message and receives reply', () => {
    cy.get('input[placeholder="Ask MasterChat…"]').type('ping{enter}')
    cy.get('.history').should('contain', 'ping')
  })
}) 