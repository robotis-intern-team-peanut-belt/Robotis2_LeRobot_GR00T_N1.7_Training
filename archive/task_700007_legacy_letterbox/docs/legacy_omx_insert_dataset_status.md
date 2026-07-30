# OMX Insert dataset status

Downloaded source:

```text
RobotisSW/Task_700007_OMX_Insert_MCAP_lerobot_v30
Hub commit: e12fb1f9612d82be662815a55fc8ceb3158f1b5d
```

The source download is preserved unchanged except for `DOWNLOAD_RECEIPT.txt`.

Verified:

- LeRobot v3.0, F2, 15 Hz
- 95 episodes, 33,730 frames, one task
- LeRobot/TorchCodec loads finite state, action, and three camera tensors
- State is 22-D; action is 19-D
- Head is 1280×720; wrist cameras are 640×480
- Natural-language instruction exists in `meta/tasks.parquet` and annotations

Not training-ready for GR00T N1.7:

1. Derive exact 16-D arm/gripper state and action features.
2. Remove mobile action and head/lift/mobile state from the policy contract.
3. Produce three identical 640×480 camera tensors; initially letterbox the head view.
4. Rewrite task metadata in current LeRobot's indexed-task format. Current loading returns `"0"` instead of the instruction because the task sentence is a column, not the DataFrame index expected by LeRobot 0.6.1.
5. Review frame-reuse records, annotations, episode quality, and a session-level validation split before training.

Do not mutate this downloaded source. Create an immutable, versioned training derivative under `datasets/`.
