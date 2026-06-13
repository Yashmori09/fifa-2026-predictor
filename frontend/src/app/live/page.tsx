import LiveClient from "./LiveClient";
import { getLiveData } from "@/lib/live-data";

// Revalidate the page every 60 seconds (ISR).
// Client also polls every 60s when any match is IN_PLAY (see LiveClient).
export const revalidate = 60;

export default async function LivePage() {
  let data;
  try {
    data = await getLiveData();
  } catch (err) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[60vh] px-4 gap-3">
        <p className="text-pink text-sm font-mono">Live data temporarily unavailable</p>
        <p className="text-secondary text-xs text-center max-w-[400px]">
          {err instanceof Error ? err.message : "Unknown error"} · Check back in a few minutes.
        </p>
      </div>
    );
  }
  return <LiveClient data={data} />;
}
