#Cosmos DB uses custom RBAC for the data plane, distinct from the control plane RBACs in the portal. This script will grant a user full access to the Cosmos DB account. Generally intended for managed identities or service principals.


# Set variables
$resourceGroupName = "djg-sandbox"
$accountName = "djg-cosmos"
$roleName = "CosmosDBFullAccess"
$principalId = "ae210e30-0b06-4004-b410-add63c352b66"


# Check if role definition already exists
$existingRoleDefinitionId = $(az cosmosdb sql role definition list --account-name $accountName --resource-group $resourceGroupName --query "[?roleName=='$roleName'].name" --output tsv)

if (-not $existingRoleDefinitionId) {
    # Role doesn't exist, create it
    Write-Host "Creating role '$roleName'..."
    
    # Create role definition JSON
    $roleDefinition = @"
{
    "RoleName": "$roleName",
    "Type": "CustomRole",
    "AssignableScopes": ["/"],
    "Permissions": [{
        "DataActions": [
            "Microsoft.DocumentDB/databaseAccounts/readMetadata",
            "Microsoft.DocumentDB/databaseAccounts/sqlDatabases/containers/items/*",
            "Microsoft.DocumentDB/databaseAccounts/sqlDatabases/containers/*",
            "Microsoft.DocumentDB/databaseAccounts/sqlDatabases/*"
        ]
     }]
}
"@
    
    $roleDefinition | Out-File -FilePath ./role-definition.json
    
    # Create the role
    az cosmosdb sql role definition create --account-name $accountName --resource-group $resourceGroupName --body "@role-definition.json"
    
    # Get the role definition ID of the newly created role
    $roleDefinitionId = $(az cosmosdb sql role definition list --account-name $accountName --resource-group $resourceGroupName --query "[?roleName=='$roleName'].name" --output tsv)
} else {
    # Role already exists, use its ID
    $roleDefinitionId = $existingRoleDefinitionId
    Write-Host "Role '$roleName' already exists with ID: $roleDefinitionId"
}

# Assign the role to the principal
Write-Host "Assigning role to principal: $principalId"
az cosmosdb sql role assignment create --account-name $accountName --resource-group $resourceGroupName --scope "/" --principal-id $principalId --role-definition-id $roleDefinitionId


#az cosmosdb sql role assignment create --account-name $accountName --resource-group $resourceGroupName --role-definition-id "00000000-0000-0000-0000-000000000002" --scope "/" --principal-id $principalId