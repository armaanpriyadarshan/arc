import { ChatNavbar } from "@/components/chat-navbar"
import { ChatInterface } from "@/components/chat-interface"

export default function ChatPage() {
  return (
    <div className="flex h-screen flex-col">
      <ChatNavbar />
      <ChatInterface />
    </div>
  )
}
