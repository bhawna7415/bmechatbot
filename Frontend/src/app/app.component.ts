import { Component } from '@angular/core';
import { ChatbotComponent } from './chatbot/chatbot.component';
import { RouterOutlet } from '@angular/router'; 
import { AppService } from './app.service';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [ChatbotComponent, RouterOutlet], 
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent {
  title = 'My Angular App';
  constructor(private appService: AppService) { this.appService.configure(); }
}
