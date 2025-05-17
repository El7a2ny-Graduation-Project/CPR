# CPR Module

## Fatema's Notion Notes
All notes for GP2 can be found [here](https://thrilling-replace-8f2.notion.site/GP-2-198751816986802088c7d036c7f4b927?pvs=4)

## TODO List for Real-Time CPR
- [ ] Handle the following cases:
  - [x] No correct postures.
  - [ ] The reporting of very short chunks.
  - [ ] The patient's head being on the left side.
- [ ] Test the following:
  - [ ] Frame rotation.
  - [ ] Frame flipping.
  - [ ] Correct rate and depth calculations.
- [ ] Implement the following:
  - [ ] Integrating the ping pong ball logic here for testing.

## Fatema's Notes to Self
- Formated warnings some times contain posture and metric warnings (check 2.mp4)
- This is how the x and y coordinates change in the frame
```` plain
(0,0) → X increases → (width,0)
  ↓
Y increases
  ↓
(0,height)        (width,height)
````