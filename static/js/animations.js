/* ========================================
   MaaSarthi - Scroll Animations JavaScript
   ======================================== */

document.addEventListener('DOMContentLoaded', function() {
    
    // ========================================
    // AUTO-ADD ANIMATION CLASSES TO ELEMENTS
    // ========================================
    
    const animationMap = {
        // Section headers
        '.section-header, .section-title': 'animate-fade-up',
        
        // Cards - Home page
        '.feature-card, .skill-card, .testimonial-card': 'animate-scale',
        '.step-card': 'animate-fade-up',
        '.result-card, .job-card': 'animate-fade-up',
        
        // Grid items with stagger
        '.features-grid > *, .skills-grid > *': 'animate-fade-up',
        '.steps-container > *': 'animate-fade-up',
        
        // About & Contact sections
        '#about, #contact': 'animate-fade-up',
        '.about-content, .contact-content': 'animate-fade-up',
        
        // CTA section
        '.cta-section': 'animate-scale',
        
        // Footer
        '.footer-section': 'animate-fade-up',
        
        // Forms
        '.form-container, .auth-card, .login-card, .signup-card': 'animate-scale',
        
        // Dashboard elements
        '.dashboard-nav': 'animate-fade-down',
        '.welcome-section': 'animate-fade-up',
        '.welcome-content': 'animate-fade-left',
        '.quick-stats': 'animate-fade-right',
        '.quick-stat-item': 'animate-scale',
        '.action-card': 'animate-fade-up',
        '.action-cards': 'animate-fade-up',
        '.dashboard-card': 'animate-scale',
        '.stat-card': 'animate-scale',
        '.activity-card': 'animate-fade-up',
        '.progress-card': 'animate-fade-up',
        '.dashboard-grid': 'animate-fade-up',
        '.grid-left, .grid-right': 'animate-fade-up',
        '.sidebar-card': 'animate-fade-right',
        
        // Login/Signup pages
        '.login-page, .signup-page': 'animate-fade-up',
        '.login-branding, .signup-branding': 'animate-fade-left',
        '.login-form-section, .signup-form-section': 'animate-fade-right',
        '.brand-section': 'animate-scale',
        '.login-form, .signup-form': 'animate-scale',
        
        // Form pages
        '.page-wrap': 'animate-fade-up',
        '.form-card': 'animate-scale',
        '.card': 'animate-scale',
        '.top-mini': 'animate-fade-down',
        
        // Result pages
        '.card-header': 'animate-fade-down',
        '.card-content': 'animate-fade-up',
        '.job-list': 'animate-fade-up',
        '.result-list': 'animate-fade-up',
        
        // Contact page
        '.contact-form': 'animate-scale',
        
        // Assistant page
        '#chatbox': 'animate-scale'
    };
    
    // Apply animation classes
    Object.entries(animationMap).forEach(([selector, animClass]) => {
        document.querySelectorAll(selector).forEach((el, index) => {
            if (!el.classList.contains('animate-on-scroll')) {
                el.classList.add('animate-on-scroll', animClass);
                // Add stagger delay
                if (index > 0 && index < 9) {
                    el.classList.add(`delay-${index}`);
                }
            }
        });
    });
    
    // ========================================
    // INTERSECTION OBSERVER FOR SCROLL ANIMATIONS
    // ========================================
    
    const observerOptions = {
        root: null,
        rootMargin: '0px 0px -50px 0px',
        threshold: 0.1
    };
    
    const animationObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('is-visible');
                // Optional: Stop observing after animation
                // animationObserver.unobserve(entry.target);
            }
        });
    }, observerOptions);
    
    // Observe all elements with animation class
    document.querySelectorAll('.animate-on-scroll').forEach(el => {
        animationObserver.observe(el);
    });
    
    // ========================================
    // PARALLAX EFFECT FOR HERO SECTION
    // ========================================
    
    const heroSection = document.querySelector('.hero-section');
    const heroImage = document.querySelector('.hero-image, .hero-img');
    
    if (heroSection && heroImage) {
        window.addEventListener('scroll', () => {
            const scrolled = window.pageYOffset;
            const rate = scrolled * 0.3;
            
            if (scrolled < window.innerHeight) {
                heroImage.style.transform = `translateY(${rate}px)`;
            }
        });
    }
    
    // ========================================
    // SMOOTH REVEAL FOR NAVBAR ON SCROLL
    // ========================================
    
    const navbar = document.querySelector('.navbar');
    let lastScroll = 0;
    
    if (navbar) {
        window.addEventListener('scroll', () => {
            const currentScroll = window.pageYOffset;
            
            if (currentScroll > 100) {
                navbar.style.boxShadow = '0 2px 20px rgba(0, 0, 0, 0.1)';
                navbar.style.background = 'rgba(255, 255, 255, 0.98)';
            } else {
                navbar.style.boxShadow = 'none';
                navbar.style.background = 'white';
            }
            
            // Hide/show navbar on scroll
            if (currentScroll > lastScroll && currentScroll > 500) {
                navbar.style.transform = 'translateY(-100%)';
            } else {
                navbar.style.transform = 'translateY(0)';
            }
            
            lastScroll = currentScroll;
        });
    }
    
    // ========================================
    // COUNTER ANIMATION FOR STATS
    // ========================================
    
    const animateCounter = (element, target, duration = 2000) => {
        let start = 0;
        const increment = target / (duration / 16);
        
        const updateCounter = () => {
            start += increment;
            if (start < target) {
                element.textContent = Math.floor(start).toLocaleString();
                requestAnimationFrame(updateCounter);
            } else {
                element.textContent = target.toLocaleString();
            }
        };
        
        updateCounter();
    };
    
    // Observe stat numbers
    const statNumbers = document.querySelectorAll('.stat-number, .counter');
    const counterObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const target = parseInt(entry.target.getAttribute('data-count') || entry.target.textContent);
                if (!isNaN(target)) {
                    animateCounter(entry.target, target);
                }
                counterObserver.unobserve(entry.target);
            }
        });
    }, { threshold: 0.5 });
    
    statNumbers.forEach(el => counterObserver.observe(el));
    
    // ========================================
    // TYPING EFFECT FOR HERO TEXT
    // ========================================
    
    const typingElements = document.querySelectorAll('.typing-effect');
    
    typingElements.forEach(el => {
        const text = el.textContent;
        el.textContent = '';
        el.style.visibility = 'visible';
        
        let i = 0;
        const typeWriter = () => {
            if (i < text.length) {
                el.textContent += text.charAt(i);
                i++;
                setTimeout(typeWriter, 50);
            }
        };
        
        // Start typing when element is visible
        const typingObserver = new IntersectionObserver((entries) => {
            if (entries[0].isIntersecting) {
                typeWriter();
                typingObserver.unobserve(el);
            }
        });
        
        typingObserver.observe(el);
    });
    
    // ========================================
    // RIPPLE EFFECT FOR BUTTONS
    // ========================================
    
    document.querySelectorAll('.btn, button, .cta-btn').forEach(button => {
        button.addEventListener('click', function(e) {
            const ripple = document.createElement('span');
            const rect = this.getBoundingClientRect();
            const size = Math.max(rect.width, rect.height);
            const x = e.clientX - rect.left - size / 2;
            const y = e.clientY - rect.top - size / 2;
            
            ripple.style.cssText = `
                position: absolute;
                width: ${size}px;
                height: ${size}px;
                left: ${x}px;
                top: ${y}px;
                background: rgba(255, 255, 255, 0.4);
                border-radius: 50%;
                transform: scale(0);
                animation: rippleEffect 0.6s ease-out;
                pointer-events: none;
            `;
            
            this.style.position = 'relative';
            this.style.overflow = 'hidden';
            this.appendChild(ripple);
            
            setTimeout(() => ripple.remove(), 600);
        });
    });
    
    // Add ripple keyframes
    if (!document.querySelector('#ripple-style')) {
        const style = document.createElement('style');
        style.id = 'ripple-style';
        style.textContent = `
            @keyframes rippleEffect {
                to {
                    transform: scale(4);
                    opacity: 0;
                }
            }
        `;
        document.head.appendChild(style);
    }
    
    // ========================================
    // TILT EFFECT FOR CARDS
    // ========================================
    
    const tiltCards = document.querySelectorAll('.feature-card, .skill-card, .testimonial-card');
    
    tiltCards.forEach(card => {
        card.addEventListener('mousemove', function(e) {
            const rect = this.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            const centerX = rect.width / 2;
            const centerY = rect.height / 2;
            
            const rotateX = (y - centerY) / 20;
            const rotateY = (centerX - x) / 20;
            
            this.style.transform = `perspective(1000px) rotateX(${rotateX}deg) rotateY(${rotateY}deg) translateY(-5px)`;
        });
        
        card.addEventListener('mouseleave', function() {
            this.style.transform = 'perspective(1000px) rotateX(0) rotateY(0) translateY(0)';
        });
    });
    
    // ========================================
    // SMOOTH SCROLL FOR ANCHOR LINKS
    // ========================================
    
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            const targetId = this.getAttribute('href');
            if (targetId === '#') return;
            
            const targetElement = document.querySelector(targetId);
            if (targetElement) {
                e.preventDefault();
                targetElement.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
    
    // ========================================
    // PAGE LOAD ANIMATION
    // ========================================
    
    document.body.classList.add('page-loaded');
    
    // Animate elements that should be visible on load
    setTimeout(() => {
        document.querySelectorAll('.animate-on-scroll').forEach(el => {
            const rect = el.getBoundingClientRect();
            if (rect.top < window.innerHeight && rect.bottom > 0) {
                el.classList.add('is-visible');
            }
        });
    }, 100);
    
    // ========================================
    // MAGNETIC EFFECT FOR CTA BUTTONS
    // ========================================
    
    const magneticButtons = document.querySelectorAll('.cta-btn, .hero-btns .btn');
    
    magneticButtons.forEach(button => {
        button.addEventListener('mousemove', function(e) {
            const rect = this.getBoundingClientRect();
            const x = e.clientX - rect.left - rect.width / 2;
            const y = e.clientY - rect.top - rect.height / 2;
            
            this.style.transform = `translate(${x * 0.2}px, ${y * 0.2}px)`;
        });
        
        button.addEventListener('mouseleave', function() {
            this.style.transform = 'translate(0, 0)';
        });
    });
    
    console.log('✨ MaaSarthi animations loaded!');
});
