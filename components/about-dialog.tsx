"use client"

import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Github, Linkedin, Mail, X } from "lucide-react"
import Image from "next/image"

interface AboutDialogProps {
  isOpen: boolean
  onClose: () => void
}

export default function AboutDialog({ isOpen, onClose }: AboutDialogProps) {
  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="max-w-[95vw] sm:max-w-4xl max-h-[90vh] overflow-y-auto glass-card mx-2 sm:mx-auto">
        <DialogHeader>
          <div className="flex flex-col sm:flex-row sm:items-start sm:justify-between gap-2 sm:gap-4">
            <div className="flex-1 min-w-0">
              <DialogTitle className="heading-primary text-lg sm:text-xl lg:text-2xl font-bold gradient-text leading-tight">
                Job Fraud Detection System
              </DialogTitle>
              <p className="text-xs sm:text-sm text-muted-foreground mt-1 sm:mt-2 text-body">
                Meet the developers behind this Job Fraud Detection System
              </p>
            </div>
            <Button variant="ghost" size="sm" onClick={onClose} className="flex-shrink-0 self-end sm:self-start">
              <X className="h-4 w-4" />
            </Button>
          </div>
        </DialogHeader>

        <div className="space-y-4 sm:space-y-6 mt-4 sm:mt-6">
          {/* Project Info */}
          <Card className="glass-card-dark card-copper">
            <CardHeader>
              <CardTitle className="heading-secondary text-base sm:text-lg lg:text-xl font-semibold text-center gradient-text leading-tight px-2">
                Job Fraud Detection System
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-center text-muted-foreground text-body">
                An advanced machine learning system designed to detect fraudulent job postings using ensemble models,
                natural language processing, and comprehensive feature engineering.
              </p>
            </CardContent>
          </Card>

          {/* Developers */}
          <div className="grid gap-4 sm:gap-6 grid-cols-1 md:grid-cols-2">
            {/* Usham Roy */}
            <Card className="glass-card-dark card-copper hover:scale-105 transition-transform duration-300">
              <CardHeader className="text-center">
                <div className="w-24 h-24 mx-auto mb-4 rounded-full overflow-hidden relative border-2 border-gold">
                  <Image
                    src="/images/usham-roy.jpeg"
                    alt="Usham Roy"
                    layout="fill"
                    objectFit="cover"
                    className="rounded-full"
                  />
                </div>
                <CardTitle className="heading-secondary text-base sm:text-lg lg:text-xl font-semibold gradient-text px-2">
                  Usham Roy
                </CardTitle>
                <p className="text-sm text-muted-foreground text-body">Lead Developer & ML Engineer</p>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-center justify-center space-x-4">
                  <a
                    href="https://github.com/uroy80"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex items-center space-x-2 text-sm hover:text-gold transition-colors text-accent"
                  >
                    <Github className="h-4 w-4" />
                    <span>GitHub</span>
                  </a>
                  <a
                    href="https://www.linkedin.com/in/ushamroy/"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex items-center space-x-2 text-sm hover:text-gold transition-colors text-accent"
                  >
                    <Linkedin className="h-4 w-4" />
                    <span>LinkedIn</span>
                  </a>
                </div>
                <div className="flex items-center justify-center">
                  <a
                    href="mailto:ushamroy80@gmail.com"
                    className="flex items-center space-x-2 text-sm hover:text-gold transition-colors text-accent"
                  >
                    <Mail className="h-4 w-4" />
                    <span>ushamroy80@gmail.com</span>
                  </a>
                </div>
                <div className="text-xs text-muted-foreground text-center text-body">
                  Specialized in machine learning algorithms, data preprocessing, and model optimization for fraud
                  detection systems.
                </div>
              </CardContent>
            </Card>

            {/* Anwesha Roy */}
            <Card className="glass-card-dark card-copper hover:scale-105 transition-transform duration-300">
              <CardHeader className="text-center">
                <div className="w-24 h-24 mx-auto mb-4 rounded-full overflow-hidden relative border-2 border-gold">
                  <Image
                    src="/images/anwesha-roy.jpeg"
                    alt="Anwesha Roy"
                    layout="fill"
                    objectFit="cover"
                    className="rounded-full"
                  />
                </div>
                <CardTitle className="heading-secondary text-base sm:text-lg lg:text-xl font-semibold gradient-text px-2">
                  Anwesha Roy
                </CardTitle>
                <p className="text-sm text-muted-foreground text-body">Frontend Developer & UI/UX Designer</p>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-center justify-center space-x-4">
                  <a
                    href="https://github.com/aroy80"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex items-center space-x-2 text-sm hover:text-gold transition-colors text-accent"
                  >
                    <Github className="h-4 w-4" />
                    <span>GitHub</span>
                  </a>
                  <a
                    href="https://www.linkedin.com/in/anwesharoy80/"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex items-center space-x-2 text-sm hover:text-gold transition-colors text-accent"
                  >
                    <Linkedin className="h-4 w-4" />
                    <span>LinkedIn</span>
                  </a>
                </div>
                <div className="flex items-center justify-center">
                  <a
                    href="mailto:royanweshasmx@gmail.com"
                    className="flex items-center space-x-2 text-sm hover:text-gold transition-colors text-accent"
                  >
                    <Mail className="h-4 w-4" />
                    <span>royanweshasmx@gmail.com</span>
                  </a>
                </div>
                <div className="text-xs text-muted-foreground text-center text-body">
                  Expert in React development, modern UI frameworks, and creating intuitive user experiences for complex
                  systems.
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Technologies Used */}
          <Card className="glass-card-dark card-copper">
            <CardHeader>
              <CardTitle className="heading-secondary text-lg font-semibold text-center gradient-text">
                Technologies & Frameworks
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 sm:gap-4 text-center">
                <div className="p-2 sm:p-3 rounded-lg bg-gradient-to-r from-blue-500/20 to-purple-500/20">
                  <div className="font-semibold text-gold text-sm sm:text-base">Frontend</div>
                  <div className="text-xs text-muted-foreground text-body">Next.js, React, TypeScript</div>
                </div>
                <div className="p-2 sm:p-3 rounded-lg bg-gradient-to-r from-green-500/20 to-blue-500/20">
                  <div className="font-semibold text-gold text-sm sm:text-base">ML/AI</div>
                  <div className="text-xs text-muted-foreground text-body">Python, scikit-learn, NLTK</div>
                </div>
                <div className="p-2 sm:p-3 rounded-lg bg-gradient-to-r from-purple-500/20 to-pink-500/20">
                  <div className="font-semibold text-gold text-sm sm:text-base">UI/UX</div>
                  <div className="text-xs text-muted-foreground text-body">Tailwind CSS, shadcn/ui</div>
                </div>
                <div className="p-2 sm:p-3 rounded-lg bg-gradient-to-r from-orange-500/20 to-red-500/20">
                  <div className="font-semibold text-gold text-sm sm:text-base">Data Viz</div>
                  <div className="text-xs text-muted-foreground text-body">Recharts, Matplotlib</div>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Footer */}
          <div className="text-center text-xs sm:text-sm text-muted-foreground text-body px-2">
            <p>© 2025 Job Fraud Detection System. Built with ❤️ for the Anveshan Hackathon.</p>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  )
}
