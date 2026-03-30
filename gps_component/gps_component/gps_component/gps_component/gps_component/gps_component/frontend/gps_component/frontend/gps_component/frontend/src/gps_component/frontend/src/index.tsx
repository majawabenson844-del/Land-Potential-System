import React from "react";
import { createRoot } from "react-dom/client";
import GpsComponent from "./GpsComponent";

const container = document.getElementById("root")!;
const root = createRoot(container);
root.render(<GpsComponent />);
