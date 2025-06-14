#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from core.registrar import register_app
import uvicorn

app = register_app()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
