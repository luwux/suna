"use client"

import { SidebarLeft } from "@/components/sidebar/sidebar-left"
import {
  SidebarInset,
  SidebarProvider,
} from "@/components/ui/sidebar"

interface DashboardLayoutProps {
  children: React.ReactNode
}

export default function DashboardLayout({
  children,
}: DashboardLayoutProps) {
  // Maintenance alert removed

  return (
    <SidebarProvider>
      <SidebarLeft />
      <SidebarInset>
        <div className="bg-background">
          {children}
        </div>
      </SidebarInset>
    </SidebarProvider>
  )
}