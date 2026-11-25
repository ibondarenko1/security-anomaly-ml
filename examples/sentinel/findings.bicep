targetScope = 'resourceGroup'

@description('An existing Log Analytics workspace in this resource group.')
param workspaceName string

@description('Azure region of the existing workspace.')
param workspaceLocation string

@description('Name for the direct-ingestion data collection rule.')
param ruleName string = 'security-anomaly-findings'

resource workspace 'Microsoft.OperationalInsights/workspaces@2023-09-01' existing = {
  name: workspaceName
}

var columns = [
  { name: 'TimeGenerated', type: 'datetime' }
  { name: 'Finding', type: 'dynamic' }
]

resource table 'Microsoft.OperationalInsights/workspaces/tables@2022-10-01' = {
  parent: workspace
  name: 'SecurityAnomalyFindings_CL'
  properties: {
    schema: {
      name: 'SecurityAnomalyFindings_CL'
      columns: columns
    }
  }
}

resource dcr 'Microsoft.Insights/dataCollectionRules@2024-03-11' = {
  name: ruleName
  location: workspaceLocation
  kind: 'Direct'
  properties: {
    streamDeclarations: {
      'Custom-SecurityAnomalyFindings': {
        columns: columns
      }
    }
    destinations: {
      logAnalytics: [{
        name: 'workspace'
        workspaceResourceId: workspace.id
      }]
    }
    dataFlows: [{
      streams: ['Custom-SecurityAnomalyFindings']
      destinations: ['workspace']
      transformKql: 'source'
      outputStream: 'Custom-${table.name}'
    }]
  }
}

output ruleResourceId string = dcr.id
output immutableId string = dcr.properties.immutableId
output logsIngestion string = dcr.properties.endpoints.logsIngestion
