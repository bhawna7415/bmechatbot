import { Component, ViewEncapsulation, AfterViewInit, ElementRef, OnInit, Renderer2, NgZone, SecurityContext, ChangeDetectorRef } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ChatbotService } from './chatbot.service';
import { ButtonModule } from '@syncfusion/ej2-angular-buttons';
import { DomSanitizer, SafeHtml } from '@angular/platform-browser';
import * as CryptoJS from 'crypto-js';
import { ActivatedRoute } from '@angular/router';
import { Location } from '@angular/common';

@Component({
  selector: 'app-chatbot',
  standalone: true,
  imports: [CommonModule, FormsModule, ButtonModule],
  templateUrl: './chatbot.component.html',
  styleUrls: ['./chatbot.component.css'],
  styles: [`
    :host {
      display: block;
      width: 800px !important;
      min-width: 800px !important;
    }
  `],
  encapsulation: ViewEncapsulation.None
})
export class ChatbotComponent implements OnInit, AfterViewInit {
  botAvatar: string = 'assets/images/imgb.jfif';
  userAvatar: string = 'assets/images/person.png';
  userInput: string = '';
  chatLog: Array<{ type: string, text: string | SafeHtml, time: string, isDefault?: boolean }> = [];
  loading: boolean = false;
  otp: string = '';
  authorized: boolean = false;

  constructor(
    private chatbotService: ChatbotService,
    private elementRef: ElementRef,
    private renderer: Renderer2,
    private ngZone: NgZone,
    private sanitizer: DomSanitizer,
    private route: ActivatedRoute,
    private location: Location,
    private cdr: ChangeDetectorRef
  ) {
    let psencryptedKey = this.route.snapshot.queryParamMap.get('ChatBot');
    this.location.replaceState('');
    if (psencryptedKey) {
      sessionStorage.setItem('ChatBot', psencryptedKey)
      try {
        const psdecryptedKey = CryptoJS.AES.decrypt(psencryptedKey, 'BMEAISecurity9.0').toString(CryptoJS.enc.Utf8);
        this.authorized = (psdecryptedKey === 'BatchMasterAIHelp');
      } catch (err) {
        this.authorized = false;
      }
    } else {
      psencryptedKey = sessionStorage.getItem('ChatBot');
      if (psencryptedKey) {
        const psdecryptedKey = CryptoJS.AES.decrypt(psencryptedKey, 'BMEAISecurity9.0').toString(CryptoJS.enc.Utf8);
        this.authorized = (psdecryptedKey === 'BatchMasterAIHelp');
      } else {
        this.authorized = false;
      }
    } 
  }

  ngOnInit(): void {
    this.chatLog.push({
      type: 'bot',
      text: this.sanitizer.bypassSecurityTrustHtml('I am here to assist with any queries related to BatchMaster ERP. Feel free to ask if you need any help!'),
      time: new Date().toLocaleTimeString(),
      isDefault: true
    });
  }

  ngAfterViewInit() {
    this.forceContainerWidth();
  }

   generate_otp(){
    fetch('https://kb.batchmaster.in/generate-otp.php', {
      method: 'GET',
      redirect: 'manual',
      credentials: 'omit',
      cache: 'no-store',
      headers: {
        'Accept': 'application/json'
      }
    })
    .then(response => {
          
      console.log("testing")
      const contentType = response.headers.get('content-type');
      if (contentType && contentType.includes('application/json')) {
        return response.json();
      } else {
        return response.text().then(text => {
          try {
            return JSON.parse(text);
          } catch (e) {
            throw new Error('Invalid response format: ' + text);
          }
        });
      }
    })
    .then(data => { 
      
      console.log(data);
      this.otp=data.otp;
      
    })
  }

  private forceContainerWidth() {
    const containerEl = this.elementRef.nativeElement.querySelector('.container');
    if (containerEl) {
      containerEl.style.width = '800px';
      containerEl.style.minWidth = '800px';
    }

    const glassEl = this.elementRef.nativeElement.querySelector('.glass');
    if (glassEl) {
      glassEl.style.width = '800px';
      glassEl.style.minWidth = '800px';
    }
  }

  private bindVideoLinkClicks(): void {
    const links = this.elementRef.nativeElement.querySelectorAll('.kbvideo');
    console.log('Binding video links...');
    console.log('Found links:', links.length);
    links.forEach((link: HTMLElement) => {
      if (!link.getAttribute('data-bound')) {
        this.renderer.listen(link, 'click', (event: MouseEvent) => {
          event.preventDefault();
  
          const baseUrl = link.getAttribute('href');
         
          console.log(baseUrl);
          if (baseUrl) {        
              if (this.otp) {
                console.log('OTP received:', this.otp);
                
                let finalUrl = baseUrl;
                
                if (baseUrl.includes('?')) {
                  finalUrl += `&otp=${this.otp}`;
                } else {
                  finalUrl += `?otp=${this.otp}`;
                }
                
                console.log('Opening URL:', finalUrl);
                
                window.open(finalUrl, '_blank');
                
              } else {
                console.error('OTP data not received properly:', this.otp);
                alert('Failed to generate OTP. Please try again.');
              }
          }
        });
        link.setAttribute('data-bound', 'true');
      }
    });
  }

  private scrollToBottom(): void {
    setTimeout(() => {
      const container = this.elementRef.nativeElement.querySelector('#chatLog');
      if (container) {
        container.scrollTop = container.scrollHeight;
      }
    }, 10);
  }

  sendMessage(): void {
    if (this.userInput.trim()) {
      if (this.chatLog[0].isDefault == true) {
        this.chatLog[0].isDefault = false;
      }
      const userMessage = {
        type: 'user',
        text: this.userInput,
        time: new Date().toLocaleTimeString()
      };
      this.chatLog.push(userMessage);

      const input = this.userInput;
      this.userInput = '';
      this.loading = true;

      const botMessage = {
        type: 'bot',
        text: '',
        time: new Date().toLocaleTimeString()
      };
      this.chatLog.push(botMessage);
      
      const botMessageIndex = this.chatLog.length - 1;

      let accumulatedText = '';

      this.scrollToBottom();

      this.chatbotService.getBotResponseStream(input).subscribe({
        next: (chunk) => {
          this.ngZone.run(() => {
            console.log('Processing chunk:', chunk);
            
            accumulatedText += chunk;
            
            this.chatLog[botMessageIndex].text = this.sanitizer.bypassSecurityTrustHtml(accumulatedText);
            this.generate_otp();

          setTimeout(() => {
            this.forceContainerWidth();
            this.bindVideoLinkClicks(); 
          }, 100);
            this.cdr.detectChanges();
            this.scrollToBottom();
          });
        },
        error: (err) => {
          this.ngZone.run(() => {
            console.error('Error:', err);
            this.chatLog[botMessageIndex].text = this.sanitizer.bypassSecurityTrustHtml('Oops! Something went wrong.');
            this.loading = false;
            this.cdr.detectChanges();
            this.scrollToBottom();
          });
        },
        complete: () => {
          this.ngZone.run(() => {
            console.log('Stream complete. Final accumulated text length:', accumulatedText.length);
            this.loading = false;
            this.cdr.detectChanges();
            this.scrollToBottom();
          });
        }
      });
    }
  }

  copyToClipboard(html: string | SafeHtml): void {
    const cleanText = typeof html === 'string' ? html : html.toString();
    const tempEl = document.createElement('div');
    tempEl.innerHTML = cleanText;
    navigator.clipboard.writeText(tempEl.innerText).then(() => {
      console.log('Copied:', tempEl.innerText);
    }).catch(err => {
      console.error('Failed to copy:', err);
    });
  }
}



