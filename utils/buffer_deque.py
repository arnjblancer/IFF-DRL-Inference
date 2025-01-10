from collections import deque

class CustomDeque(deque):
    def __init__(self, maxlen=None, expert_data=None):
        super().__init__(maxlen=maxlen)
        self.expert_data = expert_data

    def append(self, x):
        if len(self) >= self.maxlen:
            # 如果元素总数已经大于等于最大大小，移除最旧的元素
            self.popleft()
        super().append(x)


