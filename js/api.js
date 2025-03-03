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
    async postJson(route, json) {
        const body = new FormData();
        body.append("json", JSON.stringify(json));
        return await this.fetchJson(route, { method: "POST", body });
    }
    getLoras(force = false) {
        if (!this.getLorasPromise || force) {
            this.getLorasPromise = this.fetchJson("/loras", { cache: "no-store" });
        }
        return this.getLorasPromise;
    }
    async fetchApiJsonOrNull(route, options) {
        const response = await this.fetchJson(route, options);
        if (response.status === 200 && response.data) {
            return response.data || null;
        }
        return null;
    }
}

export const aeApi = new AeApi();
