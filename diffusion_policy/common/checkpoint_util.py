from typing import Optional, Dict
import os
import pathlib
import re

class TopKCheckpointManager:
    def __init__(self,
            save_dir,
            monitor_key: str,
            mode='min',
            k=1,
            format_str='epoch={epoch:03d}-train_loss={train_loss:.3f}.ckpt'
        ):
        assert mode in ['max', 'min']
        assert k >= 0

        self.save_dir = save_dir
        self.monitor_key = monitor_key
        self.mode = mode
        self.k = k
        self.format_str = format_str
        self.path_value_map = dict()
        self._filename_pattern = self._compile_filename_pattern(format_str)
        self._load_existing_checkpoints()

    def _compile_filename_pattern(self, format_str: str):
        pattern = re.escape(format_str)
        pattern = re.sub(r'\\\{([a-zA-Z0-9_]+):[^}]+\\\}', r'(?P<\1>[^/]+)', pattern)
        pattern = re.sub(r'\\\{([a-zA-Z0-9_]+)\\\}', r'(?P<\1>[^/]+)', pattern)
        suffix = os.path.splitext(format_str)[1]
        if suffix:
            escaped_suffix = re.escape(suffix)
            if pattern.endswith(escaped_suffix):
                pattern = pattern[: -len(escaped_suffix)] + rf"(?:_epoch\d+)?{escaped_suffix}"
        return re.compile(f"^{pattern}$")

    def _load_existing_checkpoints(self):
        save_dir = pathlib.Path(self.save_dir)
        if not save_dir.exists():
            return

        candidates = list()
        for path in save_dir.iterdir():
            if not path.is_file():
                continue
            match = self._filename_pattern.fullmatch(path.name)
            if match is None:
                continue
            raw_value = match.groupdict().get(self.monitor_key)
            if raw_value is None:
                continue
            try:
                value = float(raw_value)
            except ValueError:
                continue
            candidates.append((str(path), value))

        reverse = self.mode == 'max'
        candidates.sort(key=lambda x: x[1], reverse=reverse)
        kept = candidates[:self.k]
        self.path_value_map = dict(kept)

        for path_str, _ in candidates[self.k:]:
            try:
                os.remove(path_str)
            except FileNotFoundError:
                pass
    
    def get_ckpt_path(self, data: Dict[str, float]) -> Optional[str]:
        if self.k == 0:
            return None

        value = data[self.monitor_key]
        ckpt_path = os.path.join(
            self.save_dir, self.format_str.format(**data))
        
        if len(self.path_value_map) < self.k:
            # under-capacity
            self.path_value_map[ckpt_path] = value
            return ckpt_path
        
        # at capacity
        sorted_map = sorted(self.path_value_map.items(), key=lambda x: x[1])
        min_path, min_value = sorted_map[0]
        max_path, max_value = sorted_map[-1]

        delete_path = None
        if self.mode == 'max':
            if value > min_value:
                delete_path = min_path
        else:
            if value < max_value:
                delete_path = max_path

        if delete_path is None:
            return None
        else:
            del self.path_value_map[delete_path]
            self.path_value_map[ckpt_path] = value

            if not os.path.exists(self.save_dir):
                os.mkdir(self.save_dir)

            if os.path.exists(delete_path):
                os.remove(delete_path)
            return ckpt_path
