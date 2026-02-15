# Logging and Debugging

## Visualize with TensorBoard

You can inspect training metrics (reward curves, loss values, episode lengths) using TensorBoard. This requires installing it on your local machine.

1.  **Install TensorBoard:**
    On your local computer (do NOT run this on Greene), install the package:
    ```
    pip install tensorboard
    ```

2.  **Launch the Server:**
    Navigate to the folder where you downloaded your logs and start the server:
    ```
    # Assuming you are in the directory containing the 'logs' folder
    tensorboard --logdir ./logs
    ```

3.  **View Metrics:**
    Open your browser to the URL shown (usually `http://localhost:6006/`).

## Debugging on Burst

Burst storage is accessible only from a job running on burst, not from the burst login node. The provided scripts do not automatically synchronize error logs back to your home directory on Greene. However, you will need access to these logs to debug failed jobs. These error logs differ from the logs in the previous section.

The suggested way to inspect these logs is via the Open OnDemand web interface:

1.  Navigate to [https://ood-burst-001.hpc.nyu.edu](https://ood-burst-001.hpc.nyu.edu).
2.  Select **Files** > **Home Directory** from the top menu.
3.  You will see a list of files, including your `.err` log files.
4.  Click on any `.err` file to view its content directly in the browser.

> **Important:** Do not modify anything inside the `rloclab` folder on burst storage. This directory is managed by the job scripts, and manual changes may cause synchronization issues or job failures.