from typing import Literal, Optional

from pydantic import BaseModel


class Args(BaseModel):
    window_size: Literal[10, 30, 60]
    cms_width: Literal[2048, 8192]
    cs_rows: Optional[Literal[5, 7, 9]] = None
    cs_buckets: Optional[Literal[2048, 8192]] = None
    ams_r: Literal[16, 64, 256]
    run_all: Literal[True, False] = False
    mode: Literal['sketch', 'baseline']
