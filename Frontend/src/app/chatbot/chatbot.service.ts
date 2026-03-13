import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';
import { environment } from '../../environments/environment';
import * as CryptoJS from 'crypto-js';

@Injectable({
  providedIn: 'root'
})
export class ChatbotService {
  apiUrl = environment.apiUrl
  
  constructor() {}

  getBotResponseStream(message: string): Observable<string> {
    return new Observable<string>(observer => {
      const url = `${this.apiUrl}/query_stream`;

      let Time = new Date().toLocaleTimeString('en-US', {
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
        hour12: true,
        timeZone: 'UTC' // force GMT/UTC
      });

      const SecretKey = "BMEAISecurity9.0";//AITokenKey
      let Token: string = 'BatchMasterAIHelp$$' + Time;
      const encrypted = CryptoJS.AES.encrypt(Token, SecretKey).toString();

      fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': encrypted
        },
        body: JSON.stringify({ text: message })
      }).then(async response => {
        if (!response.body) {
          observer.error('No response body');
          return;
        }
        const reader = response.body.getReader();
        const decoder = new TextDecoder('utf-8');

        let done = false;
        while (!done) {
          const { value, done: doneReading } = await reader.read();
          done = doneReading;
          if (value) {
            const chunk = decoder.decode(value, { stream: true });
            // observer.next(chunk);
            const lines = chunk.split("\n");
            for (const line of lines) {
              if (!line.startsWith("data:")) continue;

              const data = line.replace(/^data:\s*/, "").trim();
              if (data === "[END]") {
                console.log("Stream ended");
                observer.complete();
                return;
              } else if (data.startsWith("[ERROR]")) {
                console.error("Stream error:", data);
                observer.error(data);
                return;
              } else if (data) {
                observer.next(data);
              }
            }
          }
        }
        observer.complete();
      }).catch(err => {
        observer.error(err);
      });

      // Cleanup function
      return () => {
        console.log('Stream canceled');
      };
      // console.log('Connecting to:', url);
      // const eventSource = new EventSource(url);

      // let hasReceivedData = false;

      // eventSource.onopen = (event) => {
      //   console.log('EventSource connection opened:', event);
      // };

      // eventSource.onmessage = (event) => {
      //   console.log('Received chunk at', new Date().toISOString(), ':', event.data);
      //   hasReceivedData = true;
        
      //   if (event.data === '[END]') {
      //     console.log('Stream ended');
      //     observer.complete();
      //     eventSource.close();
      //   } else if (event.data.startsWith('[ERROR]')) {
      //     console.error('Stream error:', event.data);
      //     observer.error(event.data);
      //     eventSource.close();
      //   } else if (event.data && event.data.trim()) {
      //     observer.next(event.data);
      //   } else {
      //     console.log('Skipping empty chunk:', event.data);
      //   }
      // };

      // eventSource.onerror = (error) => {
      //   console.error('EventSource error:', error);
      //   console.error('EventSource readyState:', eventSource.readyState);
        
      //   if (!hasReceivedData) {
      //     observer.error('Connection failed - no data received');
      //   } else {
      //     observer.error('Connection error occurred during streaming');
      //   }
      //   eventSource.close();
      // };

      // // Cleanup function
      // return () => {
      //   console.log('Closing EventSource');
      //   eventSource.close();
      // };
    });
  }
}


