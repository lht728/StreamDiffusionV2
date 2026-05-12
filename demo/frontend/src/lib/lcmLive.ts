import { get, writable } from 'svelte/store';


export enum LCMLiveStatus {
    CONNECTED = "connected",
    DISCONNECTED = "disconnected",
    PAUSED = "paused",
    WAIT = "wait",
    SEND_FRAME = "send_frame",
    TIMEOUT = "timeout",
}

const initStatus: LCMLiveStatus = LCMLiveStatus.DISCONNECTED;

export const lcmLiveStatus = writable<LCMLiveStatus>(initStatus);
export const streamId = writable<string | null>(null);

let websocket: WebSocket | null = null;
let userId: string | null = null;
export const lcmLiveActions = {
    async start(getSreamdata: () => any[]) {
        return new Promise((resolve, reject) => {

            try {
                // If an existing websocket exists and is open, reuse it
                if (websocket && websocket.readyState === WebSocket.OPEN) {
                    lcmLiveStatus.set(LCMLiveStatus.CONNECTED);
                    websocket.send(JSON.stringify({ status: "resume", timestamp: Date.now() }));
                    streamId.set(userId);
                    resolve({ status: "connected"});
                    return;
                } else if (websocket) {
                    // Tear down any stale instance (CONNECTING/CLOSING/CLOSED) before creating a new one
                    try {
                        websocket.onopen = null;
                        websocket.onclose = null;
                        websocket.onerror = null;
                        websocket.onmessage = null;
                        websocket.close();
                    } catch (_) { /* noop */ }
                    websocket = null;
                }

                userId = crypto.randomUUID();
                const websocketURL = `${window.location.protocol === "https:" ? "wss" : "ws"
                    }://${window.location.host}/api/ws/${userId}`;

                const ws = new WebSocket(websocketURL);
                websocket = ws;
                ws.onopen = () => {
                    console.log("Connected to websocket");
                };
                ws.onclose = () => {
                    if (websocket === ws) {
                        lcmLiveStatus.set(LCMLiveStatus.DISCONNECTED);
                    }
                    console.log("Disconnected from websocket");
                };
                ws.onerror = (err) => {
                    console.error(err);
                };
                ws.onmessage = (event) => {
                    // Ignore messages from a stale instance that has been replaced
                    if (websocket !== ws) return;
                    const data = JSON.parse(event.data);
                    switch (data.status) {
                        case "connected":
                            lcmLiveStatus.set(LCMLiveStatus.CONNECTED);
                            streamId.set(userId);
                            resolve({ status: "connected", userId });
                            break;
                        case "send_frame":
                            if (get(lcmLiveStatus) === LCMLiveStatus.PAUSED) {
                                break;
                            }
                            if (ws.readyState !== WebSocket.OPEN) {
                                break;
                            }
                            lcmLiveStatus.set(LCMLiveStatus.SEND_FRAME);
                            const streamData = getSreamdata();
                            ws.send(JSON.stringify({
                                status: "next_frame",
                                timestamp: Date.now()
                            }));
                            for (const d of streamData) {
                                this.send(d);
                            }
                            break;
                        case "wait":
                            if (get(lcmLiveStatus) === LCMLiveStatus.PAUSED) {
                                break;
                            }
                            lcmLiveStatus.set(LCMLiveStatus.WAIT);
                            break;
                        case "timeout":
                            console.log("timeout");
                            lcmLiveStatus.set(LCMLiveStatus.TIMEOUT);
                            streamId.set(null);
                            reject(new Error("timeout"));
                            break;
                        case "error":
                            console.log(data.message);
                            lcmLiveStatus.set(LCMLiveStatus.DISCONNECTED);
                            streamId.set(null);
                            reject(new Error(data.message));
                            break;
                    }
                };

            } catch (err) {
                console.error(err);
                lcmLiveStatus.set(LCMLiveStatus.DISCONNECTED);
                streamId.set(null);
                reject(err);
            }
        });
    },
    send(data: Blob | { [key: string]: any }) {
        if (websocket && websocket.readyState === WebSocket.OPEN) {
            if (data instanceof Blob) {
                websocket.send(data);
            } else {
                websocket.send(JSON.stringify(data));
            }
        } else {
            console.log("WebSocket not connected");
        }
    },
    async stop() {
        lcmLiveStatus.set(LCMLiveStatus.DISCONNECTED);
        if (websocket) {
            websocket.close();
        }
        websocket = null;
        streamId.set(null);
    },
    async pause() {
        if (websocket && websocket.readyState === WebSocket.OPEN) {
            websocket.send(JSON.stringify({ status: "pause", timestamp: Date.now() }));
        }
        lcmLiveStatus.set(LCMLiveStatus.PAUSED);
        streamId.set(null);
    },
};
