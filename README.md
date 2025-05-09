# CPR Module

## Fatema's Notion Notes
All notes for GP2 can be found [here](https://thrilling-replace-8f2.notion.site/GP-2-198751816986802088c7d036c7f4b927?pvs=4)

## Branch Overview & Pending Tasks
* **main**: Ignore.
* **sampling**: Ignore.
* **educational**:
  * Add validation + error reporting of calculated depth and rate after each mini chunk.
  * Return result in proper format (e.g., video + graph).
  * Handle tiny mini chunks caused by posture errors interrupting the mini chunk.
* **real-time**:
  * Implement sampling properly (only depth is currently correct).
  * Return text output in proper format (e.g., text message to be read).
  * Disable drawing since it will not even be returned to the app.

## General Pending Tasks
* Handle a video with 100% posture errors.