#!/bin/sh

uvicorn main:app --host $AI_HOST --port $AI_PORT --reload