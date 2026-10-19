// Pitchside on Azure: Container Apps (API + UI), Azure OpenAI (agents),
// AI Speech (read-aloud), Event Hubs (live feed transport), Log Analytics.
targetScope = 'resourceGroup'

@description('Short environment name used in resource names')
param environmentName string
param location string = resourceGroup().location
@description('Azure OpenAI model deployment used by the agents')
param modelName string = 'gpt-4.1-mini'
param modelVersion string = '2025-04-14'
@description('Container image; azd sets this on deploy')
param webImage string = 'mcr.microsoft.com/azuredocs/containerapps-helloworld:latest'

var token = toLower(uniqueString(resourceGroup().id, environmentName))
var tags = { 'azd-env-name': environmentName, app: 'pitchside' }

resource logs 'Microsoft.OperationalInsights/workspaces@2023-09-01' = {
  name: 'log-${token}'
  location: location
  tags: tags
  properties: { sku: { name: 'PerGB2018' }, retentionInDays: 30 }
}

resource registry 'Microsoft.ContainerRegistry/registries@2023-07-01' = {
  name: 'cr${token}'
  location: location
  tags: tags
  sku: { name: 'Basic' }
  properties: { adminUserEnabled: false }
}

resource openai 'Microsoft.CognitiveServices/accounts@2024-10-01' = {
  name: 'oai-${token}'
  location: location
  tags: tags
  kind: 'OpenAI'
  sku: { name: 'S0' }
  properties: { customSubDomainName: 'oai-${token}', publicNetworkAccess: 'Enabled', disableLocalAuth: true }
}

resource deployment 'Microsoft.CognitiveServices/accounts/deployments@2024-10-01' = {
  parent: openai
  name: modelName
  sku: { name: 'GlobalStandard', capacity: 30 }
  properties: { model: { format: 'OpenAI', name: modelName, version: modelVersion } }
}

resource speech 'Microsoft.CognitiveServices/accounts@2024-10-01' = {
  name: 'spch-${token}'
  location: location
  tags: tags
  kind: 'SpeechServices'
  sku: { name: 'F0' }
  properties: { publicNetworkAccess: 'Enabled' }
}

resource eventhubs 'Microsoft.EventHub/namespaces@2024-01-01' = {
  name: 'evhns-${token}'
  location: location
  tags: tags
  sku: { name: 'Basic', tier: 'Basic', capacity: 1 }
}

resource hub 'Microsoft.EventHub/namespaces/eventhubs@2024-01-01' = {
  parent: eventhubs
  name: 'match-events'
  properties: { partitionCount: 2, messageRetentionInDays: 1 }
}

resource env 'Microsoft.App/managedEnvironments@2024-03-01' = {
  name: 'cae-${token}'
  location: location
  tags: tags
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: { customerId: logs.properties.customerId, sharedKey: logs.listKeys().primarySharedKey }
    }
  }
}

resource identity 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' = {
  name: 'id-${token}'
  location: location
  tags: tags
}

// Cognitive Services OpenAI User: lets the app call the model with its managed identity (no keys).
resource openaiRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  scope: openai
  name: guid(openai.id, identity.id, '5e0bd9bd-7b93-4f28-af87-19fc36ad61bd')
  properties: {
    principalId: identity.properties.principalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '5e0bd9bd-7b93-4f28-af87-19fc36ad61bd')
  }
}

// AcrPull for the app identity.
resource acrPull 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  scope: registry
  name: guid(registry.id, identity.id, '7f951dda-4ed3-4680-a7ca-43fe172d538d')
  properties: {
    principalId: identity.properties.principalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '7f951dda-4ed3-4680-a7ca-43fe172d538d')
  }
}

resource web 'Microsoft.App/containerApps@2024-03-01' = {
  name: 'ca-web-${token}'
  location: location
  tags: union(tags, { 'azd-service-name': 'web' })
  identity: { type: 'UserAssigned', userAssignedIdentities: { '${identity.id}': {} } }
  properties: {
    managedEnvironmentId: env.id
    configuration: {
      ingress: { external: true, targetPort: 8000, transport: 'auto' }
      registries: [ { server: registry.properties.loginServer, identity: identity.id } ]
      secrets: [
        { name: 'speech-key', value: speech.listKeys().key1 }
        { name: 'eventhub-conn', value: listKeys('${eventhubs.id}/AuthorizationRules/RootManageSharedAccessKey', '2024-01-01').primaryConnectionString }
      ]
    }
    template: {
      containers: [
        {
          name: 'web'
          image: webImage
          resources: { cpu: json('1.0'), memory: '2Gi' }
          env: [
            { name: 'PITCHSIDE_LLM', value: 'azure' }
            { name: 'AZURE_OPENAI_ENDPOINT', value: openai.properties.endpoint }
            { name: 'AZURE_OPENAI_DEPLOYMENT', value: deployment.name }
            { name: 'AZURE_CLIENT_ID', value: identity.properties.clientId }
            { name: 'AZURE_SPEECH_KEY', secretRef: 'speech-key' }
            { name: 'AZURE_SPEECH_REGION', value: location }
            { name: 'EVENTHUB_CONNECTION_STRING', secretRef: 'eventhub-conn' }
            { name: 'EVENTHUB_NAME', value: hub.name }
          ]
        }
      ]
      scale: { minReplicas: 0, maxReplicas: 3 }
    }
  }
}

output AZURE_CONTAINER_REGISTRY_ENDPOINT string = registry.properties.loginServer
output WEB_URL string = 'https://${web.properties.configuration.ingress.fqdn}'
