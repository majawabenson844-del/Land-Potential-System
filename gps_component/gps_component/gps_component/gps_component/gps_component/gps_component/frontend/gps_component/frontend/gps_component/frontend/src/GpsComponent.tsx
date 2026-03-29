import React, { useEffect, useState } from "react";
import { Streamlit } from "streamlit-component-lib";

type GPSData = {
  latitude: number;
  longitude: number;
  accuracy?: number | null;
  altitude?: number | null;
  altitudeAccuracy?: number | null;
  heading?: number | null;
  speed?: number | null;
} | null;

export default function GpsComponent() {
  const [requested, setRequested] = useState(false);

  useEffect(() => {
    if (requested) return;
    setRequested(true);

    if (!navigator.geolocation) {
      Streamlit.setComponentValue(null);
      return;
    }

    navigator.geolocation.getCurrentPosition(
      (pos) => {
        const c = pos.coords;
        const data: GPSData = {
          latitude: c.latitude,
          longitude: c.longitude,
          accuracy: c.accuracy ?? null,
          altitude: (c.altitude as any) ?? null,
          altitudeAccuracy: (c.altitudeAccuracy as any) ?? null,
          heading: (c.heading as any) ?? null,
          speed: (c.speed as any) ?? null
        };
        Streamlit.setComponentValue(data);
      },
      () => {
        Streamlit.setComponentValue(null);
      },
      {
        enableHighAccuracy: true,
        timeout: 10000,
        maximumAge: 0
      }
    );
  }, [requested]);

  return <div style={{ width: 1, height: 1 }} />;
}
