from typing import Literal

from pydantic import BaseModel


class Args(BaseModel):
    window_size: Literal[10, 30, 60]
    cms_width: Literal[2048, 8192]
    cs_rows: Literal[5, 7, 9]
    cs_buckets: Literal[2048, 8192]
    ams_rows: Literal[16, 64, 256]
