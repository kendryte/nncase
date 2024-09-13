#!/bin/bash
version=$1
# Detect the latest installed .NET SDK version starting with 7.x.x
DOTNET_VERSION=$(dotnet --list-sdks | grep "^$version\." | sort -V | tail -n 1 | awk '{print $1}')

# Check if a version was found
if [ -z "$DOTNET_VERSION" ]; then
    echo "No .NET SDK version 7.x.x found. Please install .NET 7 SDK."
    exit 1
fi

# Output the detected .NET SDK version
echo "Detected .NET SDK version: $DOTNET_VERSION"

# Create or update the global.json file with the detected SDK version
dotnet new globaljson --sdk-version "$DOTNET_VERSION" --force

# Verify that the global.json has been created or updated
if [ -f "global.json" ]; then
    echo "global.json has been created/updated with .NET SDK version $DOTNET_VERSION"
    cat global.json
else
    echo "Failed to create or update global.json"
    exit 1
fi

