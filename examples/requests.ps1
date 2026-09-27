# Synthetic demo only. Do not put real applicant data in this repository.
$sample = Get-Content "$PSScriptRoot/sample_applicant.json" -Raw

Invoke-RestMethod -Method Get -Uri "http://127.0.0.1:8000/health"
Invoke-RestMethod -Method Get -Uri "http://127.0.0.1:8000/ready"
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/predict" -ContentType "application/json" -Body $sample

$batch = @{ applicants = @((ConvertFrom-Json $sample), (ConvertFrom-Json $sample)) } | ConvertTo-Json -Depth 5
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/predict/batch" -ContentType "application/json" -Body $batch
