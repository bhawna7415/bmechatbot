import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http'; 

@Injectable({
    providedIn: 'root'
})
export class AppService {

    public ServerAPIEndpoint: string = '';

    constructor(private httpClient: HttpClient) { }

    public configure() {
        this.httpClient.get("assets/apipath/apiurl.txt", { responseType: 'text' }).subscribe((response) => {
            try {
                const data = JSON.parse(response);
                this.ServerAPIEndpoint = data.apiurl;
                console.log("API Endpoint loaded");
            } catch (err) {
                console.error("Invalid JSON format in apiurl.txt", err);
            }
        });
    }
}
