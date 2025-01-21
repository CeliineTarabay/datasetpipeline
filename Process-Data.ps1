# PowerShell Script to Process CSV Files in data and Create a Combined Dataset

# Define the data directory
$dataDir = CUsersCompu TechnoDownloadsScriptdata  # Update if needed

# Define output file
$outputFile = Join-Path -Path $dataDir -ChildPath CombinedDataset.csv

# Initialize an array to hold combined data
$combinedData = @()

# Get all bio CSV files
$bioFiles = Get-ChildItem -Path $dataDir -Filter _bio.csv

foreach ($bioFile in $bioFiles) {
    # Extract the subject number using regex (e.g., '001' from '001_bio.csv')
    if ($bioFile.Name -match ^(d{3})_bio.csv$) {
        $subjectNumber = $matches[1]
        $envFileName = $subjectNumber`_env.csv
        $envFilePath = Join-Path -Path $dataDir -ChildPath $envFileName

        # Check if corresponding env file exists
        if (-not (Test-Path -Path $envFilePath)) {
            Write-Host Env file '$envFileName' not found for subject '$subjectNumber'. Skipping. -ForegroundColor Yellow
            continue
        }

        # Import bio data
        $bioData = Import-Csv -Path $bioFile.FullName

        # Read the timestamp from bio file's B2 (Assuming first data row's B column)
        # Adjust 'Timestamp' if the actual header name is different
        $bioTimestamp = $bioData[0].Timestamp

        # Exclude columns A and B from bio data
        # Replace 'ColumnA' and 'ColumnB' with actual column headers to exclude
        $bioColumnsToExclude = @(ColumnA, ColumnB)  # Update with actual column names
        $bioColumnsToKeep = $bioData  Get-Member -MemberType NoteProperty  Select-Object -ExpandProperty Name  Where-Object { $_ -notin $bioColumnsToExclude }
        $bioDataFiltered = $bioData  Select-Object $bioColumnsToKeep

        # Import env data
        $envData = Import-Csv -Path $envFilePath

        # Exclude columns A and C from env data
        # Replace 'ColumnA' and 'ColumnC' with actual column headers to exclude
        $envColumnsToExclude = @(ColumnA, ColumnC)  # Update with actual column names
        $envColumnsToKeep = $envData  Get-Member -MemberType NoteProperty  Select-Object -ExpandProperty Name  Where-Object { $_ -notin $envColumnsToExclude }
        $envDataFiltered = $envData  Select-Object $envColumnsToKeep

        # Determine the number of rows
        $bioRowCount = $bioDataFiltered.Count
        $envRowCount = $envDataFiltered.Count

        # Check if bioRowCount is 8 times envRowCount
        if ($bioRowCount -ne ($envRowCount  8)) {
            Write-Host Mismatch in bio and env row counts for subject '$subjectNumber'. Expected bio rows $($envRowCount  8), actual bio rows $bioRowCount. Skipping. -ForegroundColor Red
            continue
        }

        # Process data for each env row, assign 8 bio rows
        for ($i = 0; $i -lt $envRowCount; $i++) {
            $envRow = $envDataFiltered[$i]

            # Get the corresponding 8 bio rows
            $startBioIndex = $i  8
            $bioRows = $bioDataFiltered[$startBioIndex..($startBioIndex + 7)]

            foreach ($bioRow in $bioRows) {
                # Create a combined object
                $combinedRow = [PSCustomObject]@{
                    Subject = $subjectNumber
                }

                # Add bio data prefixed with 'Bio_'
                foreach ($property in $bioRow.PSObject.Properties) {
                    $combinedRow  Add-Member -MemberType NoteProperty -Name Bio_$($property.Name) -Value $property.Value
                }

                # Add env data prefixed with 'Env_'
                foreach ($property in $envRow.PSObject.Properties) {
                    $combinedRow  Add-Member -MemberType NoteProperty -Name Env_$($property.Name) -Value $property.Value
                }

                # Add the combined row to the array
                $combinedData += $combinedRow
            }
        }

        Write-Host Processed subject '$subjectNumber'.
    }
    else {
        Write-Host Bio file '$($bioFile.Name)' does not match the expected pattern. Skipping. -ForegroundColor Yellow
    }
}

# Export combined data to a CSV file
$combinedData  Export-Csv -Path $outputFile -NoTypeInformation

Write-Host Combined dataset created at '$outputFile'. -ForegroundColor Green
