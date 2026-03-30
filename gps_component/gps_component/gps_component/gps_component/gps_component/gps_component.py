import os
import streamlit.components.v1 as components

_component_func = components.declare_component(
    "gps_component",
    path=os.path.join(os.path.dirname(__file__), "frontend", "build"),
)

def gps_component():
    return _component_func()
