/* =========================================================
   CINEMATCH — ENHANCED JAVASCRIPT LOGIC
   Added Intersection Observer for scroll animations
========================================================== */
const API_BASE = localStorage.getItem('movieApiUrl') || 'http://127.0.0.1:8000';

const state = {
    suggestionTimer: null,
    searchController: null,
    isMenuOpen: false
};

/* =========================================================
   DOM ELEMENTS
========================================================== */
const $ = (id) => document.getElementById(id);

// Nav
const navbar = document.querySelector('.navbar');
const mobileMenuBtn = $('mobileMenuBtn');
const mobileNav = $('mobileNav');
const apiStatus = $('apiStatus');
const statusText = apiStatus.querySelector('.status-text');

// Search Area
const input = $('movieInput');
const clearBtn = $('clearBtn');
const recommendBtn = $('recommendBtn');
const suggestionsBox = $('suggestions');
const searchArea = $('searchArea');

// Sections
const welcomeSection = $('welcome');
const resultsSection = $('results');
const loadingSection = $('loading');
const errorBox = $('errorBox');

// Content Targets
const selectedHeading = $('selectedHeading');
const countText = $('countText');
const selectedMovieBox = $('selectedMovie');
const recommendationsGrid = $('recommendations');

// Modal
const detailModal = $('detailModal');
const modalClose = $('modalClose');
const modalContent = $('modalContent');

// Toast
const toast = $('toast');
const toastMessage = $('toastMessage');

/* =========================================================
   SCROLL REVEAL (INTERSECTION OBSERVER)
========================================================== */
const observerOptions = {
    root: null,
    rootMargin: '0px 0px -50px 0px',
    threshold: 0.1
};

const revealObserver = new IntersectionObserver((entries, observer) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.classList.add('active');
            // Unobserve after animating once to save performance
            observer.unobserve(entry.target);
        }
    });
}, observerOptions);

function initializeScrollAnimations() {
    document.querySelectorAll('.reveal').forEach(el => {
        revealObserver.observe(el);
    });
}

/* =========================================================
   UTILITIES
========================================================== */
const escapeHtml = (str) => String(str ?? '').replace(/[&<>'"]/g, ch => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#039;', '"': '&quot;'
}[ch]));

const getYear = (dateString) => dateString ? String(dateString).slice(0, 4) : '';
const getPoster = (title, url) => url ? url : `https://placehold.co/400x600/1c1c1e/86868b?text=${encodeURIComponent(title || 'Movie')}`;

/* =========================================================
   API COMMUNICATION
========================================================== */
async function fetchApi(path, options = {}) {
    const response = await fetch(`${API_BASE}${path}`, options);
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
        throw new Error(typeof data.detail === 'string' ? data.detail : 'Service is currently unavailable.');
    }
    return data;
}

async function checkApiHealth() {
    try {
        const data = await fetchApi('/api/health');
        if (data.models_loaded) {
            apiStatus.className = 'nav-status online';
            statusText.textContent = 'System Online';
        } else {
            apiStatus.className = 'nav-status offline';
            statusText.textContent = 'Models Offline';
        }
    } catch {
        apiStatus.className = 'nav-status offline';
        statusText.textContent = 'Offline';
    }
}

/* =========================================================
   UI CONTROLS
========================================================== */
function showToast(message, duration = 3000) {
    toastMessage.textContent = message;
    toast.classList.remove('hidden');
    
    // Trigger reflow for animation
    void toast.offsetWidth;
    toast.classList.add('show');
    
    setTimeout(() => {
        toast.classList.remove('show');
        setTimeout(() => toast.classList.add('hidden'), 400); 
    }, duration);
}

function toggleClearBtn() {
    clearBtn.classList.toggle('hidden', input.value.trim() === '');
}

function hideSuggestions() {
    suggestionsBox.classList.add('hidden');
    suggestionsBox.innerHTML = '';
}

function showError(msg) {
    errorBox.textContent = msg;
    errorBox.classList.remove('hidden');
}

function setViewState(view) {
    welcomeSection.classList.add('hidden');
    resultsSection.classList.add('hidden');
    loadingSection.classList.add('hidden');
    errorBox.classList.add('hidden');

    if (view === 'welcome') welcomeSection.classList.remove('hidden');
    if (view === 'loading') loadingSection.classList.remove('hidden');
    if (view === 'results') {
        resultsSection.classList.remove('hidden');
        // Re-initialize scroll animations for newly injected DOM nodes
        setTimeout(initializeScrollAnimations, 100);
    }
}

/* =========================================================
   SEARCH & RECOMMENDATIONS
========================================================== */
async function handleSearchInput() {
    toggleClearBtn();
    hideSuggestions();
    
    const query = input.value.trim();
    if (!query) return;

    if (state.searchController) state.searchController.abort();
    state.searchController = new AbortController();

    try {
        const data = await fetchApi(`/api/search?q=${encodeURIComponent(query)}&limit=5&use_tmdb=true`, { 
            signal: state.searchController.signal 
        });
        
        const items = data.results?.length ? data.results : data.local_results || [];
        if (!items.length) return;

        suggestionsBox.innerHTML = items.map(movie => `
            <button class="suggestion" type="button" data-title="${escapeHtml(movie.title)}">
                <span>${escapeHtml(movie.title)}</span>
                <small>${escapeHtml(getYear(movie.release_date || movie.year))}</small>
            </button>
        `).join('');
        
        suggestionsBox.classList.remove('hidden');
        
        suggestionsBox.querySelectorAll('.suggestion').forEach(btn => {
            btn.addEventListener('click', () => {
                input.value = btn.dataset.title;
                hideSuggestions();
                fetchRecommendations(btn.dataset.title);
            });
        });
    } catch (error) {
        if (error.name !== 'AbortError') hideSuggestions();
    }
}

async function fetchRecommendations(titleOverride) {
    const title = String(titleOverride || input.value).trim();
    if (!title) return input.focus();

    input.value = title;
    toggleClearBtn();
    hideSuggestions();
    setViewState('loading');

    try {
        const [recData, tmdbData] = await Promise.all([
            fetchApi(`/api/recommend?title=${encodeURIComponent(title)}&n=12&enrich_tmdb=true`),
            fetchApi(`/api/search?q=${encodeURIComponent(title)}&limit=1&use_tmdb=true`).catch(() => null)
        ]);

        const selectedMovieData = tmdbData?.results?.[0] || recData.query;
        
        selectedHeading.textContent = recData.query.title;
        countText.textContent = `${recData.count} matches`;

        renderFeaturedMovie(selectedMovieData);
        renderRecommendationsGrid(recData.recommendations || []);
        
        setViewState('results');
        
        setTimeout(() => resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' }), 100);
        showToast("Recommendations found");

    } catch (error) {
        setViewState('welcome');
        showError(error.message || 'Unable to fetch recommendations. Please try again.');
    }
}

/* =========================================================
   RENDERING
========================================================== */
function renderFeaturedMovie(movie) {
    const poster = getPoster(movie.title, movie.poster_url);
    const backdrop = movie.backdrop_url || poster;
    
    selectedMovieBox.innerHTML = `
        <div class="featured-backdrop" style="background-image:url('${escapeHtml(backdrop)}')"></div>
        <div class="featured-overlay"></div>
        <div class="featured-content">
            <h3>${escapeHtml(movie.title)}</h3>
            <p class="featured-description">${escapeHtml(movie.overview || 'A handpicked selection from the recommendation engine.')}</p>
            <div class="meta-row">
                ${movie.release_date ? `<span class="meta-tag">${escapeHtml(getYear(movie.release_date))}</span>` : ''}
                ${movie.rating ? `<span class="meta-tag">★ ${Number(movie.rating).toFixed(1)}</span>` : ''}
                ${movie.genres?.length ? `<span class="meta-tag">${escapeHtml(movie.genres.slice(0, 2).join(' · '))}</span>` : ''}
            </div>
        </div>
    `;
}

function renderRecommendationsGrid(items) {
    recommendationsGrid.innerHTML = items.map((movie) => {
        const match = Math.round(Number(movie.match_percent) || 0);
        return `
            <article class="movie-card reveal" tabindex="0" role="button" aria-label="View details for ${escapeHtml(movie.title)}">
                <div class="poster-wrap">
                    <img class="poster" src="${escapeHtml(getPoster(movie.title, movie.poster_url))}" alt="Poster" loading="lazy">
                    <div class="match-badge">${match}% Match</div>
                </div>
                <h4 class="card-title">${escapeHtml(movie.title)}</h4>
                <div class="card-meta">
                    <span>${escapeHtml(getYear(movie.release_date))}</span>
                    <span>★ ${Number(movie.rating || 0).toFixed(1)}</span>
                </div>
            </article>
        `;
    }).join('');

    recommendationsGrid.querySelectorAll('.movie-card').forEach((card, index) => {
        card.addEventListener('click', () => openMovieModal(items[index]));
        card.addEventListener('keydown', e => { if (e.key === 'Enter') openMovieModal(items[index]); });
    });
}

/* =========================================================
   MODAL LOGIC
========================================================== */
async function openMovieModal(movie) {
    detailModal.classList.remove('hidden');
    detailModal.setAttribute('aria-hidden', 'false');
    document.body.style.overflow = 'hidden';
    
    modalContent.innerHTML = `
        <div style="padding: 100px 0; text-align: center;">
            <div class="loader-ring" style="margin: 0 auto 16px;"></div>
            <p style="color: var(--text-secondary);">Loading details...</p>
        </div>
    `;

    try {
        const tmdbId = movie.tmdb_id || movie.id;
        if (!tmdbId) throw new Error("No TMDB ID");

        const details = await fetchApi(`/api/movie/${encodeURIComponent(tmdbId)}`);
        
        modalContent.innerHTML = `
            <div class="modal-hero">
                <img src="${escapeHtml(details.backdrop_url || details.poster_url || getPoster(details.title))}" alt="Backdrop">
                <div class="modal-gradient"></div>
            </div>
            <div class="modal-details">
                <h2>${escapeHtml(details.title)}</h2>
                
                <div class="modal-meta-chips">
                    ${details.release_date ? `<span class="meta-tag">${escapeHtml(getYear(details.release_date))}</span>` : ''}
                    ${details.runtime ? `<span class="meta-tag">${details.runtime} min</span>` : ''}
                    <span class="meta-tag">★ ${Number(details.rating || movie.rating).toFixed(1)}</span>
                </div>
                
                <p>${escapeHtml(details.overview || 'No overview available for this title.')}</p>
                
                ${details.cast?.length ? `
                    <h3 style="margin-bottom: 16px; font-size: 16px;">Top Cast</h3>
                    <div class="cast-grid">
                        ${details.cast.slice(0, 6).map(c => `
                            <div class="cast-item">
                                <img src="${escapeHtml(c.profile_url || getPoster(c.name))}" loading="lazy" onerror="this.src='${getPoster(c.name)}'">
                                <span>${escapeHtml(c.name)}</span>
                            </div>
                        `).join('')}
                    </div>
                ` : ''}
            </div>
        `;
    } catch {
        modalContent.innerHTML = `
            <div class="modal-details" style="padding-top: 60px;">
                <h2>${escapeHtml(movie.title)}</h2>
                <p>Additional details could not be loaded.</p>
            </div>
        `;
    }
}

function closeModal() {
    detailModal.classList.add('hidden');
    detailModal.setAttribute('aria-hidden', 'true');
    document.body.style.overflow = '';
}

/* =========================================================
   EVENT LISTENERS
========================================================== */
window.addEventListener('scroll', () => {
    if (window.scrollY > 20) {
        navbar.style.background = 'rgba(0, 0, 0, 0.6)';
    } else {
        navbar.style.background = 'rgba(0, 0, 0, 0.4)';
    }
});

mobileMenuBtn.addEventListener('click', () => {
    state.isMenuOpen = !state.isMenuOpen;
    mobileMenuBtn.setAttribute('aria-expanded', state.isMenuOpen);
    
    if (state.isMenuOpen) {
        mobileNav.classList.add('open');
        mobileNav.setAttribute('aria-hidden', 'false');
    } else {
        mobileNav.classList.remove('open');
        mobileNav.setAttribute('aria-hidden', 'true');
    }
});

mobileNav.querySelectorAll('a').forEach(link => {
    link.addEventListener('click', () => {
        state.isMenuOpen = false;
        mobileMenuBtn.setAttribute('aria-expanded', 'false');
        mobileNav.classList.remove('open');
        mobileNav.setAttribute('aria-hidden', 'true');
    });
});

input.addEventListener('input', () => {
    clearTimeout(state.suggestionTimer);
    state.suggestionTimer = setTimeout(handleSearchInput, 250);
});

input.addEventListener('keydown', e => {
    if (e.key === 'Enter') fetchRecommendations();
    if (e.key === 'Escape') hideSuggestions();
});

clearBtn.addEventListener('click', () => {
    input.value = '';
    toggleClearBtn();
    hideSuggestions();
    input.focus();
});

recommendBtn.addEventListener('click', () => fetchRecommendations());

$('newSearchBtn').addEventListener('click', () => {
    input.value = '';
    toggleClearBtn();
    window.scrollTo({ top: 0, behavior: 'smooth' });
    setTimeout(() => input.focus(), 500);
});

document.querySelectorAll('.quick-picks button').forEach(btn => {
    btn.addEventListener('click', () => fetchRecommendations(btn.dataset.title));
});

document.addEventListener('click', e => {
    if (!searchArea.contains(e.target)) hideSuggestions();
});

modalClose.addEventListener('click', closeModal);
detailModal.addEventListener('click', e => {
    if (e.target.hasAttribute('data-close-modal')) closeModal();
});
document.addEventListener('keydown', e => {
    if (e.key === 'Escape' && !detailModal.classList.contains('hidden')) closeModal();
});

/* =========================================================
   INITIALIZATION
========================================================== */
document.addEventListener('DOMContentLoaded', () => {
    toggleClearBtn();
    checkApiHealth();
    initializeScrollAnimations();
});