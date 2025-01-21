# PowerShell Script to Copy and Rename CSV Files from Env and Bio Sources

param (
    [string]$envSource = "$(Join-Path $PSScriptRoot 'envsource')",
    [string]$bioSource = "$(Join-Path $PSScriptRoot 'biosource')",
    [string]$destination = "$(Join-Path $PSScriptRoot 'data')"
)

# Define Log File Path
$logFile = Join-Path -Path $destination -ChildPath "copy_rename_log.txt"

# Function to Extract the First Two-Digit Number from a Filename
function Extract-TwoDigitNumber {
    param (
        [string]$filename
    )
    
    # Use regex to find all occurrences of two consecutive digits
    $matches = [regex]::Matches($filename, '\d{2}')
    
    if ($matches.Count -ge 1) {
        return $matches[0].Value  # Modify index if needed
    }
    else {
        return $null
    }
}

# Function to Process Files
function Process-Files {
    param (
        [string]$sourcePath,
        [string]$suffix
    )
    
    # Check if Source Directory Exists
    if (-not (Test-Path -Path $sourcePath)) {
        $message = "Source directory '$sourcePath' does not exist."
        Write-Host $message -ForegroundColor Red
        Add-Content -Path $logFile -Value "$(Get-Date) - $message"
        return
    }

    # Get all CSV files in the source directory, regardless of extension case
    $files = Get-ChildItem -Path $sourcePath -Filter "*.csv" -File -ErrorAction SilentlyContinue
    $uppercaseFiles = Get-ChildItem -Path $sourcePath -Filter "*.CSV" -File -ErrorAction SilentlyContinue
    $files += $uppercaseFiles

    if (-not $files) {
        $message = "No CSV files found in '$sourcePath'."
        Write-Host $message -ForegroundColor Yellow
        Add-Content -Path $logFile -Value "$(Get-Date) - $message"
        return
    }

    foreach ($file in $files) {
        $originalName = $file.Name
        $number = Extract-TwoDigitNumber -filename $originalName

        if ($number) {
            # Zero-pad the number to three digits
            $paddedNumber = "{0:D3}" -f [int]$number

            # Construct the new filename
            $newName = "$paddedNumber`_$suffix.csv"

            # Define source and destination full paths
            $sourceFile = $file.FullName
            $destFile = Join-Path -Path $destination -ChildPath $newName

            try {
                if (Test-Path -Path $destFile) {
                    $message = "File '$newName' already exists in destination. Skipping copy of '$originalName'."
                    Write-Host $message -ForegroundColor Yellow
                    Add-Content -Path $logFile -Value "$(Get-Date) - $message"
                    continue
                }

                Copy-Item -Path $sourceFile -Destination $destFile
                $message = "Copied '$originalName' to '$newName'."
                Write-Host $message -ForegroundColor Green
                Add-Content -Path $logFile -Value "$(Get-Date) - $message"
            }
            catch {
                $errorMsg = "Failed to copy '$originalName'. Error: $_"
                Write-Host $errorMsg -ForegroundColor Red
                Add-Content -Path $logFile -Value "$(Get-Date) - $errorMsg"
            }
        }
        else {
            $message = "No two-digit number found in '$originalName'. Skipping."
            Write-Host $message -ForegroundColor Yellow
            Add-Content -Path $logFile -Value "$(Get-Date) - $message"
        }
    }
}

# Ensure Destination Directory Exists
if (-not (Test-Path -Path $destination)) {
    try {
        New-Item -Path $destination -ItemType Directory -Force | Out-Null
        $message = "Created destination directory at '$destination'."
        Write-Host $message -ForegroundColor Green
        Add-Content -Path $logFile -Value "$(Get-Date) - $message"
    }
    catch {
        $errorMsg = "Failed to create destination directory. Error: $_"
        Write-Host $errorMsg -ForegroundColor Red
        Add-Content -Path $logFile -Value "$(Get-Date) - $errorMsg"
        exit
    }
}

# Process Environment Source Files
Process-Files -sourcePath $envSource -suffix "env"

# Process Bio Source Files
Process-Files -sourcePath $bioSource -suffix "bio"

$message = "File processing completed."
Write-Host $message -ForegroundColor Cyan
Add-Content -Path $logFile -Value "$(Get-Date) - $message"
