class AeApi {
    constructor(baseUrl) {
        this.getLorasPromise = null;
        this.baseUrl = baseUrl || "./ae";
    }
    apiURL(route) {
        return `${this.baseUrl}${route}`;
    }
    fetchApi(route, options) {
        return fetch(this.apiURL(route), options);
    }
    async fetchJson(route, options) {
        const r = await this.fetchApi(route, options);
        return await r.json();
    }
    getLoras(force = false) {
        if (!this.getLorasPromise || force) {
            this.getLorasPromise = this.fetchJson("/loras", { cache: "no-store" });
        }
        return this.getLorasPromise;
    }
    async resetNode(id) {
        let req_url = "/reset/" + id.toString();
        const r = await this.fetchApi(req_url, { method: "POST" });

        if (r.status != 200) {
            console.error("Failed to reset node!");
        }
    }
}

export const aeApi = new AeApi();
