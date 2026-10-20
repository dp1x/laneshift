LANESHIFT portable Windows ARM64 build

1. Extract this folder to %USERPROFILE%\LANESHIFT-Portable\ (or any writable folder).
2. Run Launch-LANESHIFT-GPU.cmd for the Qualcomm Adreno GPU path.
3. If that fails, run Launch-LANESHIFT-Compatibility.cmd.
4. Press F9 or click Record to start/stop a human gameplay recording.
5. Run Export-Recordings-For-VM.bat to create transfer\LANESHIFT-recordings-for-VM-<timestamp>.zip.
6. Copy that ZIP into the VM and extract under the LANESHIFT repository's imported-recordings directory.
7. Process with: python python/dataset_prep.py --telemetry <session>/telemetry.jsonl --events <session>/events.jsonl --episodes <session>/episodes.jsonl --output candidates.jsonl

All application-managed files stay inside this portable root.
